from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .physics import State, ThreePhaseMultiContinuumModel
from .solver import TimeStepRecord


def save_run(
    output_dir: str | Path,
    states: list[State],
    records: list[TimeStepRecord],
    model: ThreePhaseMultiContinuumModel,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "states.npz",
        pressure_pa=np.stack([s.pressure_pa for s in states]),
        water_saturation=np.stack([s.water_saturation for s in states]),
        gas_saturation=np.stack([s.gas_saturation for s in states]),
        well_pressure_pa=np.stack([s.well_pressure_pa for s in states]),
        node_medium=model.grid.node_medium,
        node_base_cell=model.grid.node_base_cell,
        edge_i=model.grid.edge_i,
        edge_j=model.grid.edge_j,
        edge_relation=model.grid.edge_relation,
    )
    with (output / "solver_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time_s", "dt_s", "accepted", "retries", "newton_iterations",
                "linear_iterations", "initial_residual", "final_residual",
                "elapsed_s", "initial_source", "residual_ratio",
                "warm_start_mass_ratio", "warm_start_alpha",
                "predictor_elapsed_s", "final_mass_error",
            ],
        )
        writer.writeheader()
        for item in records:
            result = item.result
            writer.writerow({
                "time_s": item.time_s,
                "dt_s": item.dt_s,
                "accepted": item.accepted,
                "retries": item.retries,
                "newton_iterations": result.iterations,
                "linear_iterations": result.linear_iterations,
                "initial_residual": result.initial_residual,
                "final_residual": result.final_residual,
                "elapsed_s": result.elapsed_s,
                "initial_source": result.initial_source,
                "residual_ratio": result.residual_ratio,
                "warm_start_mass_ratio": result.warm_start_mass_ratio,
                "warm_start_alpha": result.warm_start_alpha,
                "predictor_elapsed_s": result.predictor_elapsed_s,
                "final_mass_error": result.final_mass_error,
            })
    with (output / "newton_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time_s", "dt_s", "initial_source", "iteration",
                "residual_norm", "increment_norm", "damping", "linear_iterations",
            ],
        )
        writer.writeheader()
        for item in records:
            result = item.result
            writer.writerow({
                "time_s": item.time_s,
                "dt_s": item.dt_s,
                "initial_source": result.initial_source,
                "iteration": 0,
                "residual_norm": result.initial_residual,
                "increment_norm": "",
                "damping": 1.0,
                "linear_iterations": 0,
            })
            for record in result.records:
                writer.writerow({
                    "time_s": item.time_s,
                    "dt_s": item.dt_s,
                    "initial_source": result.initial_source,
                    "iteration": record.iteration,
                    "residual_norm": record.residual_norm,
                    "increment_norm": record.increment_norm,
                    "damping": record.damping,
                    "linear_iterations": record.linear_iterations,
                })


def summarize(records: list[TimeStepRecord]) -> dict[str, float]:
    accepted = [r for r in records if r.accepted]
    if not accepted:
        return {"accepted_steps": 0.0}
    return {
        "accepted_steps": float(len(accepted)),
        "rollbacks": float(sum(r.retries for r in records)),
        "mean_newton_iterations": float(np.mean([r.result.iterations for r in accepted])),
        "total_newton_iterations": float(sum(r.result.iterations for r in accepted)),
        "total_linear_iterations": float(sum(r.result.linear_iterations for r in accepted)),
        "online_time_s": float(sum(r.result.elapsed_s for r in accepted)),
        "predictor_time_s": float(sum(r.result.predictor_elapsed_s for r in accepted)),
        "max_final_residual": float(max(r.result.final_residual for r in accepted)),
        "max_mass_error": float(max(r.result.final_mass_error for r in accepted)),
    }
