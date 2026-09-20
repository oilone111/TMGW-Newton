from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tmgw_sim.config import load_case
from tmgw_sim.dataset import (
    TrainingSample,
    calibrate_newton_margin_relaxation,
    calibrate_physics_relaxation,
)
from tmgw_sim.graph import build_graph_arrays
from tmgw_sim.grid import build_case_grid
from tmgw_sim.physics import ThreePhaseMultiContinuumModel
from tmgw_sim.predictor import PhysicsDefectPredictor
from tmgw_sim.solver import NonlinearSolver
from tmgw_sim.training import (
    _fixed_validation_subset, _normalization, _sample_difficulty, _select_epoch_samples,
)


ROOT = Path(__file__).resolve().parents[1]


class ZeroPredictor:
    def __init__(self):
        self.calls = 0

    def predict_correction(self, model, current, previous, dt_s, extrapolated_vector):
        self.calls += 1
        return np.zeros_like(extrapolated_vector)


class FlatResponsePredictor(ZeroPredictor):
    candidate_alphas = (1.0,)
    last_relaxation = 0.5


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_case(ROOT / "configs" / "unit_5x5.yaml")
        self.model = ThreePhaseMultiContinuumModel(self.cfg, build_case_grid(self.cfg))

    def test_graph_contains_bidirectional_edges_and_measured_features(self):
        graph = build_graph_arrays(self.model, self.model.initial_state(), None, 3600.0)
        expected_edges = 2 * (self.model.grid.n_edges + len(self.cfg.wells))
        expected_nodes = self.model.grid.n_nodes + len(self.cfg.wells)
        self.assertEqual(graph.edge_index.shape[1], expected_edges)
        self.assertEqual(graph.node_features.shape[0], expected_nodes)
        self.assertEqual(graph.node_features.shape[1], 30)
        self.assertEqual(graph.well_node_count, len(self.cfg.wells))
        self.assertIn(5, graph.edge_relation)
        self.assertTrue(np.all(np.isfinite(graph.node_features)))
        self.assertTrue(np.all(np.isfinite(graph.edge_features)))

    def test_s3_residual_screening_interface(self):
        self.cfg.wells = []
        self.cfg.solver.mode = "S3"
        model = ThreePhaseMultiContinuumModel(self.cfg, build_case_grid(self.cfg))
        result = NonlinearSolver(model, self.cfg.solver, ZeroPredictor()).solve_timestep(
            model.initial_state(), 3600.0
        )
        self.assertTrue(result.converged)
        self.assertIn(result.initial_source, {"accepted_model", "residual_blend", "fallback_residual"})

    def test_v18_normalization_and_newton_weighted_sampling_contract(self):
        state = self.model.initial_state()
        graph = build_graph_arrays(self.model, state, None, 3600.0)
        correction = np.zeros(3 * self.model.n + self.model.n_wells)
        correction[: self.model.n] = 2.0e-4
        correction[self.model.n : 2 * self.model.n] = 3.0e-5
        sample = TrainingSample(
            graph=graph,
            target_correction=correction,
            target_state=self.model.pack(state),
            extrapolated_state=self.model.pack(state),
            dt_s=3600.0,
            baseline_newton_iterations=6,
            baseline_initial_residual=0.5,
        )
        normalization = _normalization([sample], "TMGW")
        self.assertEqual(normalization["version"], 11)
        self.assertEqual(len(normalization["node_mean"]), graph.node_features.shape[1])
        self.assertEqual(len(normalization["edge_mean"]), graph.edge_features.shape[1])
        self.assertTrue(np.all(np.asarray(normalization["node_scale"]) > 0.0))
        self.assertTrue(np.all(np.asarray(normalization["output_scale"]) > 0.0))
        difficulty = _sample_difficulty(sample, np.asarray(normalization["output_scale"]))
        selected = _select_epoch_samples([sample] * 8, np.arange(8.0), 4, 0.5)
        validation = _fixed_validation_subset([sample] * 8, np.arange(8.0), 4)
        self.assertTrue(np.isfinite(difficulty))
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(validation), 4)

    def test_complete_residual_teacher_and_p0_are_explicit(self):
        state = self.model.initial_state()
        extrapolated = self.model.pack(state)
        graph = build_graph_arrays(
            self.model, state, None, 3600.0, extrapolated_vector=extrapolated
        )
        relaxation, ratio = calibrate_physics_relaxation(
            self.model, state, 3600.0, extrapolated, graph
        )
        self.assertIn(relaxation, {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0})
        self.assertTrue(np.isfinite(ratio))
        self.assertLessEqual(ratio, 1.0)
        predictor = PhysicsDefectPredictor()
        correction = predictor.predict_correction(
            self.model, state, None, 3600.0, extrapolated
        )
        self.assertEqual(predictor.checkpoint_role, "p0_fixed_half_defect_baseline")
        self.assertEqual(predictor.candidate_alphas, (1.0,))
        self.assertEqual(correction.shape, extrapolated.shape)
        self.assertTrue(np.all(np.isfinite(correction)))

    def test_v18_newton_margin_teacher_is_bounded_and_safe(self):
        state = self.model.initial_state()
        extrapolated = self.model.pack(state)
        graph = build_graph_arrays(
            self.model, state, None, 3600.0, extrapolated_vector=extrapolated
        )
        relaxation, ratio = calibrate_newton_margin_relaxation(
            self.model, state, 3600.0, extrapolated, graph
        )
        self.assertIn(
            relaxation,
            {0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0},
        )
        self.assertTrue(np.isfinite(ratio))
        self.assertLessEqual(ratio, 1.0)

    def test_v18_trigger_skips_unneeded_graph_inference(self):
        self.cfg.solver.mode = "S3"
        self.cfg.solver.warm_start_trigger_residual = 1.0e-30
        predictor = ZeroPredictor()
        state = self.model.initial_state()
        result = NonlinearSolver(
            self.model, self.cfg.solver, predictor
        ).solve_timestep(state, 3600.0, previous=state, previous_dt_s=3600.0)
        self.assertTrue(result.converged)
        self.assertEqual(result.initial_source, "triggered_s1")
        self.assertEqual(predictor.calls, 0)

    def test_v18_precomputed_graph_residual_is_identical(self):
        state = self.model.initial_state()
        vector = self.model.pack(state)
        residual = self.model.residual(vector, state, 3600.0)
        direct = build_graph_arrays(
            self.model, state, None, 3600.0, extrapolated_vector=vector
        )
        reused = build_graph_arrays(
            self.model, state, None, 3600.0,
            extrapolated_vector=vector, extrapolated_residual=residual,
        )
        np.testing.assert_allclose(direct.node_features, reused.node_features)

    def test_v18_flat_response_falls_back_to_s1(self):
        self.cfg.solver.mode = "S3"
        self.cfg.solver.residual_use_max = 1.01
        predictor = FlatResponsePredictor()
        state = self.model.initial_state()
        result = NonlinearSolver(
            self.model, self.cfg.solver, predictor
        ).solve_timestep(state, 3600.0)
        self.assertTrue(result.converged)
        self.assertEqual(result.initial_source, "fallback_flat_response")
        self.assertEqual(predictor.calls, 1)


if __name__ == "__main__":
    unittest.main()
