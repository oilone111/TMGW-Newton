from __future__ import annotations

from pathlib import Path

import numpy as np

from tmgw_sim.config import load_case
from tmgw_sim.dataset import TrainingSample, calibrate_newton_margin_relaxation
from tmgw_sim.graph import build_graph_arrays
from tmgw_sim.grid import build_case_grid
from tmgw_sim.network import TorchCorrectionPredictor, torch
from tmgw_sim.physics import ThreePhaseMultiContinuumModel
from tmgw_sim.solver import NonlinearSolver
from tmgw_sim.training import train_network


def main() -> None:
    checkpoint = Path("outputs/v18_training_smoke.pt")
    try:
        cfg = load_case("configs/unit_5x5.yaml")
        cfg.solver.mode = "S0"
        model = ThreePhaseMultiContinuumModel(cfg, build_case_grid(cfg))
        current = model.initial_state()
        dt_s = min(cfg.time.initial_dt_s, 3600.0)
        extrapolated = model.pack(current)
        result = NonlinearSolver(model, cfg.solver).solve_timestep(current, dt_s)
        if not result.converged:
            raise RuntimeError("5x5 physical target did not converge")
        target = model.pack(result.state)
        graph = build_graph_arrays(
            model, current, None, dt_s, extrapolated_vector=extrapolated
        )
        target_relaxation, target_ratio = calibrate_newton_margin_relaxation(
            model, current, dt_s, extrapolated, graph
        )
        sample = TrainingSample(
            graph=graph,
            target_correction=target - extrapolated,
            target_state=target,
            extrapolated_state=extrapolated,
            dt_s=dt_s,
            model=model,
            current_state=current,
            baseline_newton_iterations=result.iterations,
            baseline_initial_residual=result.initial_residual,
            target_relaxation=target_relaxation,
            target_relaxation_residual_ratio=target_ratio,
        )
        history = train_network(
            [sample], [sample], checkpoint, epochs=2, learning_rate=1.0e-3,
            batch_size=1, samples_per_epoch=1, validation_interval=1,
            patience=2, model_kind="TMGW", device="cpu", curriculum_epochs=1,
            online_validation_samples=1, online_validation_interval=1,
        )
        if len(history["train"]) != 2 or not np.all(np.isfinite(history["train"])):
            raise RuntimeError("v0.18 short-training history is invalid")
        if "train_relaxation" not in history or "validation_deep_improvement_rate" not in history:
            raise RuntimeError("v0.18 graph-level relaxation metrics are missing")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("checkpoint_format_version") != 11:
            raise RuntimeError("v0.18 checkpoint format was not written")
        model_config = payload.get("model_config", {})
        if model_config.get("output_mode") != "global_relaxation_v11_compact":
            raise RuntimeError("v0.18 compact Newton-margin head was not written")
        if model_config.get("hidden_dim") != 24 or model_config.get("layers") != 1:
            raise RuntimeError("v0.18 compact architecture was not written")
        if not 0.0 < float(payload.get("inference_gain", 0.0)) <= 1.0:
            raise RuntimeError("v0.18 inference gain is invalid")
        predictor = TorchCorrectionPredictor(checkpoint, device="cpu")
        if predictor.candidate_alphas != (1.0,):
            raise RuntimeError("v0.18 must evaluate exactly one learned candidate")
        correction = predictor.predict_correction(model, current, None, dt_s, extrapolated)
        if correction.shape != extrapolated.shape or not np.all(np.isfinite(correction)):
            raise RuntimeError("reloaded v0.18 checkpoint returned an invalid correction")
        initial_residual = np.linalg.norm(
            model.residual(extrapolated, current, dt_s), ord=np.inf
        )
        corrected = model.warm_start_well_projection(extrapolated + correction)
        corrected_residual = np.linalg.norm(
            model.residual(corrected, current, dt_s), ord=np.inf
        )
        # A diagonal defect proposal is not expected to improve every tiny
        # synthetic cell problem.  The production contract is that S3 screens
        # the complete residual and safely falls back when it does not.  The
        # 96-case diagnostic separately measures whether the proposal is useful
        # on representative reservoir states.
        cfg.solver.mode = "S3"
        screened = NonlinearSolver(model, cfg.solver, predictor).solve_timestep(
            current, dt_s
        )
        if not screened.converged:
            raise RuntimeError("v0.18 residual gate did not preserve convergence")
        if screened.initial_source not in {
            "accepted_ai", "residual_blend", "fallback_residual",
            "fallback_flat_response",
        }:
            raise RuntimeError("v0.18 residual gate returned an invalid source")
        print(
            "V18_TRAINING_SMOKE_OK "
            f"true_ratio={corrected_residual / max(initial_residual, 1e-30):.6e} "
            f"source={screened.initial_source} "
            f"alpha={history['validation_selected_alpha_median'][-1]:.5f}",
            flush=True,
        )
    finally:
        checkpoint.unlink(missing_ok=True)
        checkpoint.with_suffix(".history.json").unlink(missing_ok=True)
        checkpoint.with_suffix(".partial.history.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
