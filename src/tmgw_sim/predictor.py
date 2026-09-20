from __future__ import annotations

from pathlib import Path

import numpy as np

from .graph import build_graph_arrays


class PhysicsDefectPredictor:
    """Deterministic P0 baseline used to isolate the learned TMGW gain."""

    checkpoint_role = "p0_fixed_half_defect_baseline"
    candidate_alphas = (1.0,)
    inference_gain = 1.0

    def __init__(self, relaxation: float = 0.5):
        self.relaxation = float(relaxation)
        self.last_relaxation = self.relaxation
        self.precomputed_residual = None

    def predict_correction(
        self, model, current, previous, dt_s, extrapolated_vector,
    ) -> np.ndarray:
        graph = build_graph_arrays(
            model, current, previous, dt_s,
            extrapolated_vector=extrapolated_vector,
            extrapolated_residual=self.precomputed_residual,
        )
        self.precomputed_residual = None
        n = graph.reservoir_node_count
        nw = graph.well_node_count
        return self.relaxation * np.concatenate([
            graph.node_features[:n, 26],
            graph.node_features[:n, 27],
            graph.node_features[:n, 28],
            graph.node_features[n : n + nw, 29],
        ])


def load_correction_predictor(checkpoint: str | Path, device: str = "cpu"):
    """Load either the current PyTorch checkpoint or a legacy JAX checkpoint."""
    checkpoint = Path(checkpoint)
    if checkpoint.suffix.lower() == ".npz":
        from .jax_network import JaxCorrectionPredictor

        return JaxCorrectionPredictor(checkpoint)
    from .network import TorchCorrectionPredictor

    return TorchCorrectionPredictor(checkpoint, device=device)
