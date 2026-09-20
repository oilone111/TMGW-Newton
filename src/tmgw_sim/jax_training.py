from __future__ import annotations

import random
import json
from pathlib import Path

import numpy as np

from .dataset import TrainingSample
from .jax_network import (
    apply_tmgw,
    init_tmgw_parameters,
    jax,
    jnp,
    load_jax_checkpoint,
    save_jax_checkpoint,
)


def _target_matrix(sample: TrainingSample):
    graph = sample.graph
    n, nw = graph.reservoir_node_count, graph.well_node_count
    correction = sample.target_correction
    target = np.zeros((n + nw, 4), dtype=np.float32)
    target[:n, 0] = correction[:n]
    target[:n, 1] = correction[n : 2 * n]
    target[:n, 2] = correction[2 * n : 3 * n]
    if nw:
        target[n:, 3] = correction[3 * n :]
    return target


def _loss_arrays(
    params, node_features, edge_index, edge_features, edge_relation, target,
    n: int, nw: int, use_threshold_gate: bool,
):
    pred = apply_tmgw(
        params, node_features, edge_index, edge_features, edge_relation,
        use_threshold_gate,
    )
    pore_weight = 10.0 ** node_features[:n, 19]
    pore_weight = pore_weight / jnp.sum(pore_weight)
    reservoir_pred = pred[:n, :3]
    reservoir_pred = reservoir_pred - jnp.sum(
        pore_weight[:, None] * reservoir_pred, axis=0, keepdims=True
    )
    gas_active = (jnp.max(jnp.abs(target[:n, 2])) > 1.0e-12).astype(pred.dtype)
    reservoir_pred = reservoir_pred.at[:, 2].multiply(gas_active)
    pred = pred.at[:n, :3].set(reservoir_pred)
    if nw:
        pred = pred.at[n:, 3].multiply(node_features[n:, 20])
    target_rms = jnp.sqrt(jnp.mean(target[:n, :3] ** 2, axis=0))
    correction_scale = jnp.maximum(
        target_rms, jnp.asarray([1.0e-4, 2.0e-5, 1.0e-5])
    )
    reservoir_loss = jnp.mean(
        ((pred[:n, :3] - target[:n, :3]) / correction_scale) ** 2
    )
    well_scale = (
        jnp.maximum(jnp.sqrt(jnp.mean(target[n:, 3] ** 2)), 1.0e-4)
        if nw else 1.0
    )
    well_loss = (
        jnp.mean(((pred[n:, 3] - target[n:, 3]) / well_scale) ** 2)
        if nw else 0.0
    )
    state_loss = reservoir_loss + well_loss

    src, dst = edge_index
    gate = edge_features[:, 4] if use_threshold_gate else jnp.ones(edge_features.shape[0])
    normalized_log = jnp.clip(edge_features[:, 0] - jnp.max(edge_features[:, 0]), -12.0, 0.0)
    conductance = 10.0 ** normalized_log
    pred_pressure = jnp.concatenate([pred[:n, 0], pred[n:, 3]])
    target_pressure = jnp.concatenate([target[:n, 0], target[n:, 3]])
    q_pred = conductance * gate * (pred_pressure[src] - pred_pressure[dst])
    q_target = conductance * gate * (target_pressure[src] - target_pressure[dst])
    div_pred = jnp.zeros((pred.shape[0],)).at[dst].add(q_pred)
    div_target = jnp.zeros((pred.shape[0],)).at[dst].add(q_target)
    residual_loss = jnp.mean((div_pred - div_target) ** 2)

    water_mass = (jnp.sum(pore_weight * pred[:n, 1]) - jnp.sum(pore_weight * target[:n, 1])) ** 2
    gas_mass = (jnp.sum(pore_weight * pred[:n, 2]) - jnp.sum(pore_weight * target[:n, 2])) ** 2
    mass_loss = water_mass + gas_mass
    bound_loss = jnp.mean(jax.nn.relu(jnp.abs(pred[:n, 1:3]) - 0.1) ** 2)
    total = state_loss + 0.2 * residual_loss + 0.5 * mass_loss + 0.1 * bound_loss
    return total, (state_loss, residual_loss, mass_loss, bound_loss)


def _loss_components(params, sample: TrainingSample, use_threshold_gate: bool):
    graph = sample.graph
    return _loss_arrays(
        params,
        jnp.asarray(graph.node_features),
        jnp.asarray(graph.edge_index),
        jnp.asarray(graph.edge_features),
        jnp.asarray(graph.edge_relation),
        jnp.asarray(_target_matrix(sample)),
        graph.reservoir_node_count,
        graph.well_node_count,
        use_threshold_gate,
    )


def _batch_loss(
    params, node_features, edge_index, edge_features, edge_relation, target,
    n: int, nw: int, use_threshold_gate: bool,
):
    losses, components = jax.vmap(
        _loss_arrays,
        in_axes=(None, 0, 0, 0, 0, 0, None, None, None),
    )(
        params, node_features, edge_index, edge_features, edge_relation, target,
        n, nw, use_threshold_gate,
    )
    return jnp.mean(losses), tuple(jnp.mean(value) for value in components)


def _stack_batch(samples: list[TrainingSample]):
    return (
        jnp.asarray(np.stack([sample.graph.node_features for sample in samples])),
        jnp.asarray(np.stack([sample.graph.edge_index for sample in samples])),
        jnp.asarray(np.stack([sample.graph.edge_features for sample in samples])),
        jnp.asarray(np.stack([sample.graph.edge_relation for sample in samples])),
        jnp.asarray(np.stack([_target_matrix(sample) for sample in samples])),
    )


def _zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_scale(tree, value):
    return jax.tree_util.tree_map(lambda x: x * value, tree)


def _global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x * x) for x in leaves))


def train_jax_network(
    train_samples: list[TrainingSample],
    validation_samples: list[TrainingSample],
    output_checkpoint: str | Path,
    epochs: int = 300,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-5,
    patience: int = 30,
    batch_size: int = 4,
    samples_per_epoch: int = 0,
    validation_interval: int = 5,
    seed: int = 20260818,
    model_kind: str = "TMGW",
    initial_checkpoint: str | Path | None = None,
) -> dict[str, list[float]]:
    if jax is None:
        raise ImportError("JAX is required for CPU training")
    if not train_samples:
        raise ValueError("training sample list is empty")
    random.seed(seed)
    np.random.seed(seed)
    first = train_samples[0].graph
    model_config = {
        "node_dim": first.node_features.shape[1],
        "edge_dim": first.edge_features.shape[1],
        "hidden_dim": 64,
        "layers": 3,
        "n_relations": 1 if model_kind.upper() == "GNN" else 6,
        "use_threshold_gate": model_kind.upper() == "TMGW",
    }
    if initial_checkpoint is None:
        params = init_tmgw_parameters(
            seed, model_config["node_dim"], model_config["edge_dim"],
            model_config["hidden_dim"], model_config["layers"], model_config["n_relations"],
        )
    else:
        params, checkpoint_config, _ = load_jax_checkpoint(initial_checkpoint)
        if checkpoint_config != model_config:
            raise ValueError("resume checkpoint architecture does not match the requested model")
    moment_1, moment_2 = _zeros_like(params), _zeros_like(params)
    beta_1, beta_2, epsilon = 0.9, 0.999, 1.0e-8
    step = 0
    history = {
        "train": [], "validation": [], "state": [], "residual": [],
        "mass": [], "bound": [], "learning_rate": [],
    }
    best_loss, stale = float("inf"), 0
    value_and_grad = jax.jit(
        jax.value_and_grad(_batch_loss, has_aux=True),
        static_argnums=(6, 7, 8),
    )
    validation_loss_fn = jax.jit(_batch_loss, static_argnums=(6, 7, 8))

    for epoch in range(epochs):
        random.shuffle(train_samples)
        epoch_samples = (
            random.sample(train_samples, samples_per_epoch)
            if 0 < samples_per_epoch < len(train_samples)
            else train_samples
        )
        total_loss = np.zeros(5, dtype=float)
        for batch_start in range(0, len(epoch_samples), batch_size):
            batch = epoch_samples[batch_start : batch_start + batch_size]
            graph = batch[0].graph
            arrays = _stack_batch(batch)
            (loss, components), gradient = value_and_grad(
                params, *arrays,
                graph.reservoir_node_count,
                graph.well_node_count,
                model_config["use_threshold_gate"],
            )
            total_loss += len(batch) * np.asarray([loss, *components], dtype=float)
            norm = _global_norm(gradient)
            gradient = _tree_scale(gradient, jnp.minimum(1.0, 1.0 / (norm + 1.0e-12)))
            cosine = 0.5 * (1.0 + np.cos(np.pi * epoch / max(epochs - 1, 1)))
            lr = 1.0e-5 + (learning_rate - 1.0e-5) * cosine
            step += 1
            moment_1 = jax.tree_util.tree_map(
                lambda m, g: beta_1 * m + (1.0 - beta_1) * g, moment_1, gradient
            )
            moment_2 = jax.tree_util.tree_map(
                lambda v, g: beta_2 * v + (1.0 - beta_2) * g * g, moment_2, gradient
            )
            params = jax.tree_util.tree_map(
                lambda p, m, v: p - lr * (
                    (m / (1.0 - beta_1 ** step))
                    / (jnp.sqrt(v / (1.0 - beta_2 ** step)) + epsilon)
                    + weight_decay * p
                ),
                params, moment_1, moment_2,
            )

        means = total_loss / len(epoch_samples)
        for key, value in zip(("train", "state", "residual", "mass", "bound"), means):
            history[key].append(float(value))
        history["learning_rate"].append(float(lr))
        evaluate_validation = epoch == 0 or (epoch + 1) % max(validation_interval, 1) == 0
        validation_values = []
        if evaluate_validation:
            for batch_start in range(0, len(validation_samples), batch_size):
                batch = validation_samples[batch_start : batch_start + batch_size]
                graph = batch[0].graph
                value, _ = validation_loss_fn(
                    params, *_stack_batch(batch),
                    graph.reservoir_node_count,
                    graph.well_node_count,
                    model_config["use_threshold_gate"],
                )
                validation_values.extend([float(value)] * len(batch))
        validation_loss = (
            float(np.mean(validation_values)) if validation_values
            else history["validation"][-1] if history["validation"]
            else float(means[0])
        )
        history["validation"].append(validation_loss)
        if evaluate_validation:
            if validation_loss < best_loss:
                best_loss, stale = validation_loss, 0
                save_jax_checkpoint(
                    output_checkpoint, params, model_config,
                    {"best_validation_loss": best_loss, "seed": seed, "model_kind": model_kind.upper()},
                )
            else:
                stale += 1
                if stale >= patience:
                    break
        Path(output_checkpoint).with_suffix(".partial.history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"epoch={epoch + 1} train={means[0]:.6e} "
                f"validation={validation_loss:.6e} best={best_loss:.6e} stale={stale}",
                flush=True,
            )
    return history
