from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tmgw_sim.config import load_case
from tmgw_sim.diagnostics import save_run, summarize
from tmgw_sim.grid import build_case_grid
from tmgw_sim.physics import ThreePhaseMultiContinuumModel
from tmgw_sim.predictor import PhysicsDefectPredictor, load_correction_predictor
from tmgw_sim.solver import NonlinearSolver, run_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact blind-check report for a TMGW checkpoint")
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/checkpoint_check"))
    args = parser.parse_args()

    cfg = load_case(args.config)
    cfg.solver.mode = "S3"
    model = ThreePhaseMultiContinuumModel(cfg, build_case_grid(cfg))
    predictor = load_correction_predictor(args.checkpoint)
    initial = model.initial_state()
    predictor.predict_correction(model, initial, None, cfg.time.initial_dt_s, model.pack(initial))
    states, records = run_schedule(
        NonlinearSolver(model, cfg.solver, predictor), initial,
        cfg.time.total_time_s, cfg.time.initial_dt_s, cfg.time.min_dt_s,
        cfg.time.max_dt_s, cfg.time.adaptive,
    )
    save_run(args.output, states, records, model)
    accepted = [item for item in records if item.accepted]
    report = summarize(records)
    report["checkpoint_role"] = str(getattr(predictor, "checkpoint_role", "unknown"))
    report["checkpoint_inference_gain"] = float(getattr(predictor, "inference_gain", 1.0))
    report["initial_source_counts"] = dict(Counter(
        item.result.initial_source for item in accepted
    ))
    ratios = [
        item.result.residual_ratio for item in accepted
        if item.result.residual_ratio is not None
    ]
    report["median_warm_start_residual_ratio"] = float(np.median(ratios)) if ratios else None
    alphas = [
        item.result.warm_start_alpha for item in accepted
        if item.result.warm_start_alpha is not None
    ]
    report["median_warm_start_alpha"] = float(np.median(alphas)) if alphas else None
    counts = report["initial_source_counts"]
    n_steps = max(len(accepted), 1)
    report["direct_ai_acceptance_rate"] = counts.get("accepted_ai", 0) / n_steps
    report["ai_used_rate"] = (
        counts.get("accepted_ai", 0) + counts.get("residual_blend", 0)
    ) / n_steps
    report["ai_trigger_skip_rate"] = counts.get("triggered_s1", 0) / n_steps
    report["fallback_rate"] = sum(
        value for key, value in counts.items()
        if key.startswith("fallback")
    ) / n_steps
    if report["median_warm_start_residual_ratio"] is not None:
        report["median_initial_residual_reduction_percent"] = 100.0 * (
            1.0 - report["median_warm_start_residual_ratio"]
        )
    report["predictor_time_share"] = report["predictor_time_s"] / max(report["online_time_s"], 1.0e-30)

    # S1 is the correct online baseline for an AI correction applied on top of
    # temporal extrapolation.  The supplied stored run remains the independent
    # S0 high-fidelity reference for final-state accuracy.
    cfg_s1 = load_case(args.config)
    cfg_s1.solver.mode = "S1"
    model_s1 = ThreePhaseMultiContinuumModel(cfg_s1, build_case_grid(cfg_s1))
    states_s1, records_s1 = run_schedule(
        NonlinearSolver(model_s1, cfg_s1.solver), model_s1.initial_state(),
        cfg_s1.time.total_time_s, cfg_s1.time.initial_dt_s, cfg_s1.time.min_dt_s,
        cfg_s1.time.max_dt_s, cfg_s1.time.adaptive,
    )
    s1 = summarize(records_s1)
    report["s1_reference_steps"] = s1["accepted_steps"]
    report["s1_mean_newton_iterations"] = s1["mean_newton_iterations"]
    report["s1_total_linear_iterations"] = s1["total_linear_iterations"]
    report["s1_online_time_s"] = s1["online_time_s"]
    report["online_speedup_vs_s1"] = s1["online_time_s"] / max(report["online_time_s"], 1.0e-30)
    report["mean_newton_reduction_vs_s1_percent"] = 100.0 * (
        1.0 - report["mean_newton_iterations"] / max(s1["mean_newton_iterations"], 1.0e-30)
    )
    report["same_accepted_step_count_vs_s1"] = len(accepted) == len([
        item for item in records_s1 if item.accepted
    ])
    report["pressure_relative_l2_vs_s1"] = float(
        np.linalg.norm(states[-1].pressure_pa - states_s1[-1].pressure_pa)
        / max(np.linalg.norm(states_s1[-1].pressure_pa), 1.0e-30)
    )
    report["water_saturation_mae_vs_s1"] = float(
        np.mean(np.abs(states[-1].water_saturation - states_s1[-1].water_saturation))
    )
    report["gas_saturation_mae_vs_s1"] = float(
        np.mean(np.abs(states[-1].gas_saturation - states_s1[-1].gas_saturation))
    )

    # P0 is the deterministic 0.5-times diagonal physics-defect proposal that
    # produced the promising v0.14 result.  It is deliberately evaluated as a
    # separate baseline so a fixed physical rule can never be reported as a
    # learned TMGW gain.
    cfg_p0 = load_case(args.config)
    cfg_p0.solver.mode = "S3"
    model_p0 = ThreePhaseMultiContinuumModel(cfg_p0, build_case_grid(cfg_p0))
    p0_predictor = PhysicsDefectPredictor(relaxation=0.5)
    states_p0, records_p0 = run_schedule(
        NonlinearSolver(model_p0, cfg_p0.solver, p0_predictor),
        model_p0.initial_state(), cfg_p0.time.total_time_s,
        cfg_p0.time.initial_dt_s, cfg_p0.time.min_dt_s,
        cfg_p0.time.max_dt_s, cfg_p0.time.adaptive,
    )
    save_run(args.output / "P0_fixed_physics", states_p0, records_p0, model_p0)
    p0 = summarize(records_p0)
    accepted_p0 = [item for item in records_p0 if item.accepted]
    p0_ratios = [
        item.result.residual_ratio for item in accepted_p0
        if item.result.residual_ratio is not None
    ]
    report["p0_fixed_physics"] = {
        **p0,
        "median_warm_start_residual_ratio": (
            float(np.median(p0_ratios)) if p0_ratios else None
        ),
        "initial_source_counts": dict(Counter(
            item.result.initial_source for item in accepted_p0
        )),
        "pressure_relative_l2_vs_s1": float(
            np.linalg.norm(states_p0[-1].pressure_pa - states_s1[-1].pressure_pa)
            / max(np.linalg.norm(states_s1[-1].pressure_pa), 1.0e-30)
        ),
    }
    report["mean_newton_reduction_vs_p0_percent"] = 100.0 * (
        1.0 - report["mean_newton_iterations"]
        / max(p0["mean_newton_iterations"], 1.0e-30)
    )
    report["linear_reduction_vs_p0_percent"] = 100.0 * (
        1.0 - report["total_linear_iterations"]
        / max(p0["total_linear_iterations"], 1.0e-30)
    )
    report["online_speedup_vs_p0"] = (
        p0["online_time_s"] / max(report["online_time_s"], 1.0e-30)
    )
    report["adaptive_step_count_note"] = (
        "Adaptive runs may contain different numbers of accepted steps for S1, P0, and S3. "
        "Accuracy is assessed at the common final time using the converged state, final residual, "
        "and mass conservation; a different step count is not treated as an accuracy failure."
    )
    if args.reference_run:
        reference = np.load(args.reference_run / "states.npz")
        p_ref = reference["pressure_pa"][-1]
        report["archived_pressure_relative_l2"] = float(
            np.linalg.norm(states[-1].pressure_pa - p_ref) / max(np.linalg.norm(p_ref), 1.0e-30)
        )
        report["archived_water_saturation_mae"] = float(
            np.mean(np.abs(states[-1].water_saturation - reference["water_saturation"][-1]))
        )
        with (args.reference_run / "solver_log.csv").open("r", encoding="utf-8") as handle:
            reference_rows = [row for row in csv.DictReader(handle) if row["accepted"].lower() == "true"]
        reference_steps = len(reference_rows)
        reference_newton = sum(int(row["newton_iterations"]) for row in reference_rows)
        reference_linear = sum(int(row["linear_iterations"]) for row in reference_rows)
        reference_time = sum(float(row["elapsed_s"]) for row in reference_rows)
        report["archived_reference_steps"] = reference_steps
        report["archived_reference_newton_iterations"] = reference_newton
        report["archived_reference_linear_iterations"] = reference_linear
        report["archived_reference_online_time_s"] = reference_time
        report["same_accepted_step_count_vs_archive"] = len(accepted) == reference_steps
        if report["same_accepted_step_count_vs_archive"]:
            report["archived_newton_reduction_percent"] = 100.0 * (
                1.0 - sum(item.result.iterations for item in accepted) / max(reference_newton, 1)
            )
            report["archived_linear_reduction_percent"] = 100.0 * (
                1.0 - sum(item.result.linear_iterations for item in accepted) / max(reference_linear, 1)
            )
    accuracy_ok = (
        report.get("pressure_relative_l2_vs_s1", 0.0) <= 1.0e-5
        and report.get("water_saturation_mae_vs_s1", 0.0) <= 1.0e-5
        and report.get("gas_saturation_mae_vs_s1", 0.0) <= 1.0e-5
        and report["rollbacks"] == 0
        and report.get("max_final_residual", float("inf"))
        <= 1.05 * cfg.solver.nonlinear_tolerance
        and report.get("max_mass_error", float("inf"))
        <= 1.05 * cfg.solver.mass_tolerance
    )
    acceleration_ok = (
        report.get("mean_newton_reduction_vs_s1_percent", 0.0) >= 5.0
        and report.get("online_speedup_vs_s1", 0.0) > 1.0
        and report["fallback_rate"] <= 0.20
        and report.get("median_warm_start_residual_ratio", 1.0) < 0.90
    )
    learned_role = report["checkpoint_role"] in {
        "learned_tmgw_global_relaxation",
        "learned_tmgw_newton_margin",
        "learned_tmgw_compact_newton_margin",
    }
    learning_contribution_ok = learned_role and (
        report.get("mean_newton_reduction_vs_p0_percent", 0.0) > 0.0
        or (
            abs(report.get("mean_newton_reduction_vs_p0_percent", 0.0))
            <= 1.0e-12
            and report.get("linear_reduction_vs_p0_percent", 0.0) > 0.0
        )
    )
    report["accuracy_check"] = "PASS" if accuracy_ok else "FAIL"
    report["acceleration_check"] = "PASS" if acceleration_ok else "NEEDS_WORK"
    report["learning_contribution_check"] = (
        "PASS" if learning_contribution_ok else "NEEDS_WORK"
    )
    report["recommended_next_task"] = (
        "run_41x41_benchmark"
        if accuracy_ok and acceleration_ok and learning_contribution_ok
        else "adjust_or_retrain_tmgw"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoint_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
