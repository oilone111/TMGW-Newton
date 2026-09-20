from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--hgnn-checkpoint", type=Path)
    parser.add_argument("--tmgw-checkpoint", type=Path)
    parser.add_argument("--fixed-dt", action="store_true")
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    args = parser.parse_args()
    variants: list[tuple[str, str, object | None]] = [
        ("S0", "S0", None),
        ("S1", "S1", None),
        ("P0", "S3", PhysicsDefectPredictor(relaxation=0.5)),
    ]
    if args.hgnn_checkpoint:
        variants.append(("S2", "S2", load_correction_predictor(args.hgnn_checkpoint)))
    if args.tmgw_checkpoint:
        variants.append(("S3", "S3", load_correction_predictor(args.tmgw_checkpoint)))
    summaries = {}
    final_states = {}
    for name, solver_mode, predictor in variants:
        repeats = max(int(args.timing_repeats), 1)
        repeated_summaries = []
        representative = None
        for repeat in range(repeats):
            cfg = load_case(args.config)
            cfg.solver.mode = solver_mode
            if args.fixed_dt:
                cfg.time.adaptive = False
            model = ThreePhaseMultiContinuumModel(cfg, build_case_grid(cfg))
            initial_state = model.initial_state()
            if predictor is not None:
                predictor.predict_correction(
                    model, initial_state, None, cfg.time.initial_dt_s,
                    model.pack(initial_state),
                )
            states, records = run_schedule(
                NonlinearSolver(model, cfg.solver, predictor), initial_state,
                cfg.time.total_time_s, cfg.time.initial_dt_s, cfg.time.min_dt_s,
                cfg.time.max_dt_s, cfg.time.adaptive,
            )
            repeated_summaries.append(summarize(records))
            if repeat == 0:
                representative = (states, records, model)
        assert representative is not None
        states, records, model = representative
        summaries[name] = dict(repeated_summaries[0])
        time_runs = [item["online_time_s"] for item in repeated_summaries]
        predictor_runs = [item["predictor_time_s"] for item in repeated_summaries]
        summaries[name]["online_time_s"] = float(np.median(time_runs))
        summaries[name]["predictor_time_s"] = float(np.median(predictor_runs))
        summaries[name]["online_time_s_runs"] = [float(value) for value in time_runs]
        summaries[name]["online_time_s_std"] = float(np.std(time_runs))
        summaries[name]["timing_repeats"] = repeats
        summaries[name]["solver_mode"] = solver_mode
        summaries[name]["checkpoint_role"] = str(
            getattr(predictor, "checkpoint_role", "none")
        )
        accepted_records = [item for item in records if item.accepted]
        source_counts = Counter(
            item.result.initial_source for item in accepted_records
        )
        summaries[name]["initial_source_counts"] = dict(source_counts)
        accepted_count = max(len(accepted_records), 1)
        summaries[name]["ai_trigger_skip_rate"] = float(
            source_counts.get("triggered_s1", 0) / accepted_count
        )
        summaries[name]["graph_inference_steps"] = int(
            len(accepted_records) - source_counts.get("triggered_s1", 0)
            if predictor is not None else 0
        )
        summaries[name]["predictor_time_share"] = float(
            summaries[name]["predictor_time_s"]
            / max(summaries[name]["online_time_s"], 1.0e-30)
        )
        final_states[name] = states[-1]
        save_run(args.output / name, states, records, model)

    reference = final_states["S0"]
    p_norm = max(np.linalg.norm(reference.pressure_pa), 1e-30)
    for mode, state in final_states.items():
        summaries[mode]["pressure_relative_l2"] = float(
            np.linalg.norm(state.pressure_pa - reference.pressure_pa) / p_norm
        )
        summaries[mode]["water_saturation_mae"] = float(
            np.mean(np.abs(state.water_saturation - reference.water_saturation))
        )
        summaries[mode]["gas_saturation_mae"] = float(
            np.mean(np.abs(state.gas_saturation - reference.gas_saturation))
        )
        summaries[mode]["well_pressure_relative_l2"] = float(
            np.linalg.norm(state.well_pressure_pa - reference.well_pressure_pa)
            / max(np.linalg.norm(reference.well_pressure_pa), 1e-30)
        )
    base_time = summaries["S0"].get("online_time_s", np.nan)
    s1_time = summaries["S1"].get("online_time_s", np.nan)
    s1_newton = summaries["S1"].get("mean_newton_iterations", np.nan)
    s1_linear = summaries["S1"].get("total_linear_iterations", np.nan)
    s1_steps = summaries["S1"].get("accepted_steps", np.nan)
    for mode in summaries:
        elapsed = summaries[mode].get("online_time_s", np.nan)
        summaries[mode]["speedup_vs_S0"] = float(base_time / elapsed) if elapsed > 0 else np.nan
        summaries[mode]["speedup_vs_S1"] = float(s1_time / elapsed) if elapsed > 0 else np.nan
        mean_newton = summaries[mode].get("mean_newton_iterations", np.nan)
        summaries[mode]["newton_reduction_vs_S1_percent"] = float(
            100.0 * (1.0 - mean_newton / s1_newton)
        ) if np.isfinite(s1_newton) and s1_newton > 0 else np.nan
        linear = summaries[mode].get("total_linear_iterations", np.nan)
        summaries[mode]["linear_reduction_vs_S1_percent"] = float(
            100.0 * (1.0 - linear / s1_linear)
        ) if np.isfinite(s1_linear) and s1_linear > 0 else np.nan
        summaries[mode]["same_accepted_steps_as_S1"] = bool(
            summaries[mode].get("accepted_steps") == s1_steps
        )
    if "S3" in summaries:
        s3 = summaries["S3"]
        p0 = summaries["P0"]
        s3["newton_reduction_vs_P0_percent"] = float(
            100.0 * (1.0 - s3["mean_newton_iterations"] / p0["mean_newton_iterations"])
        )
        s3["linear_reduction_vs_P0_percent"] = float(
            100.0 * (1.0 - s3["total_linear_iterations"] / p0["total_linear_iterations"])
        )
        s3["speedup_vs_P0"] = float(p0["online_time_s"] / s3["online_time_s"])
        accuracy_pass = bool(
            s3["pressure_relative_l2"] <= 1.0e-5
            and s3["water_saturation_mae"] <= 1.0e-5
            and s3["gas_saturation_mae"] <= 1.0e-5
            and s3["same_accepted_steps_as_S1"]
        )
        acceleration_pass = bool(
            s3["newton_reduction_vs_S1_percent"] > 0.0
            and s3["linear_reduction_vs_S1_percent"] > 0.0
            and s3["speedup_vs_S1"] > 1.0
        )
        contribution_pass = bool(
            s3["newton_reduction_vs_P0_percent"] > 0.0
            or (
                abs(s3["newton_reduction_vs_P0_percent"]) <= 1.0e-12
                and s3["linear_reduction_vs_P0_percent"] > 0.0
            )
        )
        s3["accuracy_check"] = "PASS" if accuracy_pass else "FAIL"
        s3["acceleration_check"] = "PASS" if acceleration_pass else "NEEDS_WORK"
        s3["learning_contribution_check"] = (
            "PASS" if contribution_pass else "NEEDS_WORK"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summaries, indent=2, ensure_ascii=False)
    (args.output / "summary.json").write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
