from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tmgw_sim.config import load_case
from tmgw_sim.diagnostics import save_run, summarize
from tmgw_sim.grid import build_case_grid
from tmgw_sim.physics import ThreePhaseMultiContinuumModel
from tmgw_sim.predictor import load_correction_predictor
from tmgw_sim.solver import NonlinearSolver, run_schedule


def run_variant(config: Path, checkpoint: Path, mode: str, output: Path, fixed_dt: bool):
    cfg = load_case(config)
    cfg.solver.mode = mode
    if fixed_dt:
        cfg.time.adaptive = False
    model = ThreePhaseMultiContinuumModel(cfg, build_case_grid(cfg))
    predictor = load_correction_predictor(checkpoint)
    initial = model.initial_state()
    predictor.predict_correction(model, initial, None, cfg.time.initial_dt_s, model.pack(initial))
    states, records = run_schedule(
        NonlinearSolver(model, cfg.solver, predictor), initial,
        cfg.time.total_time_s, cfg.time.initial_dt_s, cfg.time.min_dt_s,
        cfg.time.max_dt_s, cfg.time.adaptive,
    )
    save_run(output, states, records, model)
    metrics = summarize(records)
    accepted = [item for item in records if item.accepted]
    fallback = sum(
        item.result.initial_source.startswith("fallback") for item in accepted
    )
    metrics["fallback_rate"] = fallback / max(len(accepted), 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Chapter 4 module ablation")
    parser.add_argument("config", type=Path)
    parser.add_argument("--gnn-checkpoint", type=Path, required=True)
    parser.add_argument("--hgnn-checkpoint", type=Path, required=True)
    parser.add_argument("--tmgw-checkpoint", type=Path, required=True)
    parser.add_argument("--fixed-dt", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/ablation_41x41"))
    args = parser.parse_args()
    variants = [
        ("Single-relation graph", args.gnn_checkpoint, "S3"),
        ("Without threshold gate", args.hgnn_checkpoint, "S3"),
        ("Without safety gate", args.tmgw_checkpoint, "S2"),
        ("Full TMGW-Newton", args.tmgw_checkpoint, "S3"),
    ]
    rows = []
    for index, (name, checkpoint, mode) in enumerate(variants):
        metrics = run_variant(
            args.config, checkpoint, mode,
            args.output / f"variant_{index + 1}", args.fixed_dt,
        )
        rows.append({"variant": name, **metrics})
        print(name, metrics, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant", "mean_newton_iterations", "online_time_s", "fallback_rate",
        "accepted_steps", "rollbacks", "total_linear_iterations",
        "predictor_time_s", "max_final_residual", "max_mass_error",
    ]
    with (args.output / "ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    main()
