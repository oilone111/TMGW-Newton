from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .graph import build_graph_arrays
from .physics import State, ThreePhaseMultiContinuumModel

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None


def _linear(key, input_dim: int, output_dim: int) -> dict:
    limit = np.sqrt(6.0 / max(input_dim + output_dim, 1))
    return {
        "w": jax.random.uniform(key, (input_dim, output_dim), minval=-limit, maxval=limit),
        "b": jnp.zeros((output_dim,)),
    }


def _mlp(keys, input_dim: int, output_dim: int, hidden_dim: int) -> dict:
    return {
        "a": _linear(keys[0], input_dim, hidden_dim),
        "b": _linear(keys[1], hidden_dim, output_dim),
    }


def _apply_linear(params: dict, x):
    return x @ params["w"] + params["b"]


def _apply_mlp(params: dict, x):
    return _apply_linear(params["b"], jax.nn.silu(_apply_linear(params["a"], x)))


def init_tmgw_parameters(
    seed: int,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int = 64,
    layers: int = 3,
    n_relations: int = 6,
) -> dict:
    if jax is None:
        raise ImportError("JAX is required for the CPU TMGW backend")
    key = jax.random.PRNGKey(seed)
    keys = iter(jax.random.split(key, 4096))
    params = {
        "node_encoder": _mlp([next(keys), next(keys)], node_dim, hidden_dim, hidden_dim),
        "edge_encoders": {},
        "layers": {},
        "decoder": _mlp([next(keys), next(keys)], hidden_dim, 4, hidden_dim),
    }
    # A correction network should start from the safe temporal-extrapolation
    # baseline.  Zero-initialising only the final projection gives exactly zero
    # correction at epoch 0 without suppressing the learned latent features.
    params["decoder"]["b"]["w"] = jnp.zeros_like(params["decoder"]["b"]["w"])
    params["decoder"]["b"]["b"] = jnp.zeros_like(params["decoder"]["b"]["b"])
    for relation in range(n_relations):
        params["edge_encoders"][f"r{relation}"] = _mlp(
            [next(keys), next(keys)], edge_dim, hidden_dim, hidden_dim
        )
    for layer_index in range(layers):
        layer = {"messages": {}}
        for relation in range(n_relations):
            layer["messages"][f"r{relation}"] = _mlp(
                [next(keys), next(keys)], 3 * hidden_dim, hidden_dim, hidden_dim
            )
        for gate_name in ("z", "r", "n"):
            layer[f"{gate_name}_x"] = _linear(next(keys), hidden_dim, hidden_dim)
            layer[f"{gate_name}_h"] = _linear(next(keys), hidden_dim, hidden_dim)
        layer["norm_scale"] = jnp.ones((hidden_dim,))
        layer["norm_shift"] = jnp.zeros((hidden_dim,))
        params["layers"][f"l{layer_index}"] = layer
    return params


def apply_tmgw(
    params: dict,
    node_features,
    edge_index,
    edge_features,
    edge_relation,
    use_threshold_gate: bool = True,
):
    if len(params["edge_encoders"]) == 1:
        edge_relation = jnp.zeros_like(edge_relation)
    if not use_threshold_gate:
        edge_features = edge_features.at[:, 4].set(1.0)
        edge_features = edge_features.at[:, 5].set(1.0)
    h = _apply_mlp(params["node_encoder"], node_features)
    edge_hidden = jnp.zeros((edge_features.shape[0], h.shape[1]), dtype=h.dtype)
    for relation_index, encoder in enumerate(params["edge_encoders"].values()):
        encoded = _apply_mlp(encoder, edge_features)
        mask = (edge_relation == relation_index).astype(h.dtype)[:, None]
        edge_hidden = edge_hidden + mask * encoded

    src, dst = edge_index
    for layer in params["layers"].values():
        aggregate = jnp.zeros_like(h)
        for relation_index, message_params in enumerate(layer["messages"].values()):
            message_input = jnp.concatenate([h[src], h[dst], edge_hidden], axis=-1)
            message = _apply_mlp(message_params, message_input)
            mask = (edge_relation == relation_index).astype(h.dtype)[:, None]
            aggregate = aggregate.at[dst].add(mask * message)
        z = jax.nn.sigmoid(
            _apply_linear(layer["z_x"], aggregate) + _apply_linear(layer["z_h"], h)
        )
        r = jax.nn.sigmoid(
            _apply_linear(layer["r_x"], aggregate) + _apply_linear(layer["r_h"], h)
        )
        candidate = jnp.tanh(
            _apply_linear(layer["n_x"], aggregate) + _apply_linear(layer["n_h"], r * h)
        )
        updated = (1.0 - z) * candidate + z * h
        residual = h + updated
        mean = jnp.mean(residual, axis=-1, keepdims=True)
        variance = jnp.mean((residual - mean) ** 2, axis=-1, keepdims=True)
        h = (residual - mean) / jnp.sqrt(variance + 1.0e-5)
        h = h * layer["norm_scale"] + layer["norm_shift"]
    raw = _apply_mlp(params["decoder"], h)
    return jnp.stack([
        raw[:, 0],
        0.1 * jnp.tanh(raw[:, 1]),
        0.1 * jnp.tanh(raw[:, 2]),
        raw[:, 3],
    ], axis=-1)


def _flatten_params(tree: dict, prefix: str = "") -> dict[str, np.ndarray]:
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}__{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(value, name))
        else:
            flat[name] = np.asarray(value)
    return flat


def _unflatten_params(flat: dict[str, np.ndarray]) -> dict:
    root: dict = {}
    for name, value in flat.items():
        cursor = root
        parts = name.split("__")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = jnp.asarray(value)
    return root


def save_jax_checkpoint(path: str | Path, params: dict, model_config: dict, metadata: dict) -> None:
    payload = _flatten_params(params)
    payload["checkpoint_config_json"] = np.asarray(json.dumps(model_config))
    payload["checkpoint_metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(Path(path), **payload)


def load_jax_checkpoint(path: str | Path) -> tuple[dict, dict, dict]:
    arrays = np.load(Path(path), allow_pickle=False)
    model_config = json.loads(str(arrays["checkpoint_config_json"]))
    metadata = json.loads(str(arrays["checkpoint_metadata_json"]))
    flat = {
        key: arrays[key]
        for key in arrays.files
        if not key.startswith("checkpoint_")
    }
    return _unflatten_params(flat), model_config, metadata


class JaxCorrectionPredictor:
    def __init__(self, checkpoint: str | Path):
        if jax is None:
            raise ImportError("JAX is required for this checkpoint")
        self.params, self.model_config, self.metadata = load_jax_checkpoint(checkpoint)
        self.use_threshold_gate = bool(self.model_config.get("use_threshold_gate", True))
        self._forward = jax.jit(
            lambda params, nodes, edges, features, relations: apply_tmgw(
                params, nodes, edges, features, relations, self.use_threshold_gate
            )
        )

    def predict_correction(
        self,
        model: ThreePhaseMultiContinuumModel,
        current: State,
        previous: State | None,
        dt_s: float,
        extrapolated_vector: np.ndarray,
    ) -> np.ndarray:
        graph = build_graph_arrays(
            model, current, previous, dt_s,
            extrapolated_vector=extrapolated_vector,
        )
        prediction = np.asarray(self._forward(
            self.params,
            jnp.asarray(graph.node_features),
            jnp.asarray(graph.edge_index),
            jnp.asarray(graph.edge_features),
            jnp.asarray(graph.edge_relation),
        ))
        n, nw = graph.reservoir_node_count, graph.well_node_count
        pressure = prediction[:n, 0]
        water = prediction[:n, 1]
        gas = prediction[:n, 2]
        pore_weight = model._porosity * model.grid.node_volume
        pore_weight = pore_weight / np.sum(pore_weight)
        # The learned correction redistributes the extrapolated state; its
        # domain-average inventory change is supplied by the physical time
        # extrapolation and wells.  Removing the learned uniform mode prevents
        # a small neural bias from creating a large global mass imbalance.
        pressure = pressure - np.sum(pore_weight * pressure)
        water = water - np.sum(pore_weight * water)
        gas = gas - np.sum(pore_weight * gas)
        # With no mobile gas initially and no gas source in the governing
        # equations, the exact gas correction is zero.  Enforcing this known
        # invariant prevents an inactive output channel from perturbing Newton.
        gas_inactive = (
            np.max(np.abs(current.gas_saturation - model.cfg.fluid.sgc)) <= 1.0e-12
            and (
                previous is None
                or np.max(np.abs(previous.gas_saturation - model.cfg.fluid.sgc)) <= 1.0e-12
            )
        )
        if gas_inactive:
            gas = np.zeros_like(gas)
        well = prediction[n : n + nw, 3].copy()
        for well_index, cfg_well in enumerate(model.cfg.wells):
            if cfg_well.control == "bhp":
                well[well_index] = 0.0
        return np.concatenate([pressure, water, gas, well])
