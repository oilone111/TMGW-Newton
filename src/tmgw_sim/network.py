from __future__ import annotations

from pathlib import Path

import numpy as np

from .graph import build_graph_arrays
from .physics import State, ThreePhaseMultiContinuumModel

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only on installations without AI extras
    torch = None
    nn = None


if nn is not None:
    class MLP(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.net(x)


    class RelationMessageLayer(nn.Module):
        def __init__(self, hidden_dim: int, edge_dim: int, n_relations: int = 6):
            super().__init__()
            self.messages = nn.ModuleList([
                MLP(2 * hidden_dim + edge_dim, hidden_dim, hidden_dim)
                for _ in range(n_relations)
            ])
            self.update = nn.GRUCell(hidden_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, h, edge_index, edge_features, edge_relation):
            src, dst = edge_index
            aggregate = torch.zeros_like(h)
            for relation, message_mlp in enumerate(self.messages):
                mask = edge_relation == relation
                if not torch.any(mask):
                    continue
                relation_src = src[mask]
                relation_dst = dst[mask]
                message_input = torch.cat(
                    [h[relation_src], h[relation_dst], edge_features[mask]], dim=-1
                )
                message = message_mlp(message_input)
                aggregate.index_add_(0, relation_dst, message)
            updated = self.update(aggregate, h)
            return self.norm(h + updated)


    class TMGWNetwork(nn.Module):
        def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            hidden_dim: int = 64,
            layers: int = 3,
            n_relations: int = 6,
            use_threshold_gate: bool = True,
            output_mode: str = "global_relaxation_v11_compact",
        ):
            super().__init__()
            self.use_threshold_gate = use_threshold_gate
            self.output_mode = output_mode
            self.node_encoder = MLP(node_dim, hidden_dim, hidden_dim)
            self.edge_encoders = nn.ModuleList([
                MLP(edge_dim, hidden_dim, hidden_dim) for _ in range(n_relations)
            ])
            self.layers = nn.ModuleList([
                RelationMessageLayer(hidden_dim, hidden_dim, n_relations)
                for _ in range(layers)
            ])
            if output_mode in {
                "global_relaxation_v9", "global_relaxation_v10",
                "global_relaxation_v11_compact",
            }:
                # One graph represents one time step. Mean/max pooling retains
                # both the domain-wide state and the strongest local nonlinear
                # feature, while a single scalar preserves the coherent
                # approximate-Newton defect direction.
                # Mean/max pooling alone is invariant to graph size, so a
                # 21x21 graph and a 41x41 graph with similar feature
                # distributions could be indistinguishable.  v0.18 appends a
                # bounded log node-count feature to make scale transfer
                # learnable without changing the coherent scalar output.
                self.global_decoder = MLP(2 * hidden_dim + 1, 1, hidden_dim)
                nn.init.zeros_(self.global_decoder.net[-1].weight)
                nn.init.zeros_(self.global_decoder.net[-1].bias)
                self.decoders = nn.ModuleList()
            else:
                self.global_decoder = None
                self.decoders = nn.ModuleList([
                    MLP(hidden_dim, 1, hidden_dim) for _ in range(4)
                ])
                for decoder in self.decoders:
                    nn.init.zeros_(decoder.net[-1].weight)
                    nn.init.zeros_(decoder.net[-1].bias)

        def encode_edges(self, features, relation):
            encoded = torch.zeros(
                (features.shape[0], self.layers[0].update.input_size),
                dtype=features.dtype,
                device=features.device,
            )
            for r, encoder in enumerate(self.edge_encoders):
                mask = relation == r
                if torch.any(mask):
                    encoded[mask] = encoder(features[mask])
            return encoded

        def forward(self, node_features, edge_index, edge_features, edge_relation):
            if len(self.edge_encoders) == 1:
                edge_relation = torch.zeros_like(edge_relation)
            h = self.node_encoder(node_features)
            edge_hidden = self.encode_edges(edge_features, edge_relation)
            for layer in self.layers:
                h = layer(h, edge_index, edge_hidden, edge_relation)
            if self.output_mode in {
                "global_relaxation_v9", "global_relaxation_v10",
                "global_relaxation_v11_compact",
            }:
                graph_scale = torch.log1p(
                    torch.as_tensor(
                        float(h.shape[0]), dtype=h.dtype, device=h.device
                    )
                ).reshape(1) / 10.0
                pooled = torch.cat([
                    torch.mean(h, dim=0), torch.amax(h, dim=0), graph_scale,
                ], dim=-1)
                return torch.sigmoid(self.global_decoder(pooled)).reshape(1)
            raw = torch.cat([decoder(h) for decoder in self.decoders], dim=-1)
            if self.output_mode not in {
                "normalized_v2", "normalized_v3", "normalized_v4",
                "residual_conditioned_v5", "newton_margin_v6",
                "physics_multiplier_v7",
                "relaxation_teacher_v8",
                "global_relaxation_v9",
                "global_relaxation_v10",
                "global_relaxation_v11_compact",
            }:
                raise ValueError(f"unsupported network output mode: {self.output_mode}")
            # Legacy v0.14-v0.15 checkpoints predict node-wise multipliers.
            if self.output_mode in {"physics_multiplier_v7", "relaxation_teacher_v8"}:
                return torch.sigmoid(raw)
            # Bound the dimensionless correction to prevent rare nodes
            # from producing a physically catastrophic Newton initial state.
            if self.output_mode in {
                "normalized_v4", "residual_conditioned_v5", "newton_margin_v6",
            }:
                raw = 4.0 * torch.tanh(raw / 4.0)
            # Component-specific
            # physical scales stored in the checkpoint are applied outside the
            # network in both training and inference.
            return raw


class TorchCorrectionPredictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        if torch is None:
            raise ImportError("install the 'ai' optional dependencies to use TMGWNetwork")
        payload = torch.load(Path(checkpoint), map_location=device, weights_only=True)
        if payload.get("checkpoint_format_version") not in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}:
            raise ValueError(
                "checkpoint is from v0.8 or earlier; use a current PyTorch model"
            )
        self.device = torch.device(device)
        self.model = TMGWNetwork(**payload["model_config"]).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        normalization = payload.get("normalization")
        if not isinstance(normalization, dict):
            raise ValueError("checkpoint does not contain normalization statistics")
        self.node_mean = torch.as_tensor(
            normalization["node_mean"], dtype=torch.float32, device=self.device
        )
        self.node_scale = torch.as_tensor(
            normalization["node_scale"], dtype=torch.float32, device=self.device
        )
        self.edge_mean = torch.as_tensor(
            normalization["edge_mean"], dtype=torch.float32, device=self.device
        )
        self.edge_scale = torch.as_tensor(
            normalization["edge_scale"], dtype=torch.float32, device=self.device
        )
        self.output_scale = torch.as_tensor(
            normalization["output_scale"], dtype=torch.float32, device=self.device
        )
        # v0.18 uses one residual check for the learned graph-level scalar.
        # Earlier candidate strategies remain loadable for diagnostics.
        self.inference_gain = float(payload.get("inference_gain", 1.0))
        self.checkpoint_role = str(payload.get("checkpoint_role", "unknown"))
        self.last_relaxation = float("nan")
        self.precomputed_residual = None
        version = payload.get("checkpoint_format_version")
        self.candidate_alphas = (
            (1.0,) if version in {9, 10, 11} else
            (1.0, 0.75, 1.25) if version == 8 else
            (2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
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
            extrapolated_residual=self.precomputed_residual,
        )
        self.precomputed_residual = None
        edge_features = graph.edge_features.copy()
        if not self.model.use_threshold_gate:
            edge_features[:, 4:6] = 1.0
        with torch.no_grad():
            prediction_scaled = self.model(
                (
                    torch.as_tensor(graph.node_features, device=self.device)
                    - self.node_mean
                ) / self.node_scale,
                torch.as_tensor(graph.edge_index, device=self.device),
                (
                    torch.as_tensor(edge_features, device=self.device)
                    - self.edge_mean
                ) / self.edge_scale,
                torch.as_tensor(graph.edge_relation, device=self.device),
            )
            if self.model.output_mode in {
                "global_relaxation_v9", "global_relaxation_v10",
                "global_relaxation_v11_compact",
            }:
                physics_defect = torch.as_tensor(
                    graph.node_features[:, 26:30],
                    dtype=prediction_scaled.dtype,
                    device=self.device,
                )
                prediction = (
                    prediction_scaled.reshape(1, 1) * physics_defect
                    * self.inference_gain
                ).cpu().numpy()
                self.last_relaxation = float(
                    prediction_scaled.reshape(-1)[0].detach().cpu()
                ) * self.inference_gain
            elif self.model.output_mode in {"physics_multiplier_v7", "relaxation_teacher_v8"}:
                physics_defect = torch.as_tensor(
                    graph.node_features[:, 26:30],
                    dtype=prediction_scaled.dtype,
                    device=self.device,
                )
                prediction = (
                    prediction_scaled * physics_defect * self.inference_gain
                ).cpu().numpy()
            else:
                prediction = (
                    prediction_scaled * self.output_scale * self.inference_gain
                ).cpu().numpy()
        n = graph.reservoir_node_count
        nw = graph.well_node_count
        reservoir = prediction[:n].copy()
        gas_inactive = (
            np.max(np.abs(current.gas_saturation - model.cfg.fluid.sgc)) <= 1.0e-12
            and (
                previous is None
                or np.max(np.abs(previous.gas_saturation - model.cfg.fluid.sgc)) <= 1.0e-12
            )
        )
        if gas_inactive:
            reservoir[:, 2] = 0.0
        well = prediction[n : n + nw, 3].copy()
        for well_index, cfg_well in enumerate(model.cfg.wells):
            if cfg_well.control == "bhp":
                well[well_index] = 0.0
        return np.concatenate([
            reservoir[:, 0],
            reservoir[:, 1],
            reservoir[:, 2],
            well,
        ])
