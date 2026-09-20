from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from tmgw_sim.config import load_case
from tmgw_sim.diagnostics import save_run, summarize
from tmgw_sim.graph import build_graph_arrays
from tmgw_sim.grid import build_case_grid
from tmgw_sim.physics import State, ThreePhaseMultiContinuumModel
from tmgw_sim.solver import (
    NonlinearSolver,
    SolverResult,
    TimeStepRecord,
    run_schedule,
)


DEFAULT_ALPHAS = (
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
    1.125,
    1.25,
    1.5,
    1.75,
    2.0,
)


class FixedDefectPredictor:
    """Apply one prescribed scalar to the same physical defect as P0/TMGW."""

    checkpoint_role = "v18_oracle_fixed_scalar"

    def __init__(self, relaxation: float):
        self.relaxation = float(relaxation)
        self.last_relaxation = self.relaxation

    def predict_correction(
        self,
        model: ThreePhaseMultiContinuumModel,
        current: State,
        previous: State | None,
        dt_s: float,
        extrapolated_vector: np.ndarray,
    ) -> np.ndarray:
        graph = build_graph_arrays(
            model,
            current,
            previous,
            dt_s,
            extrapolated_vector=extrapolated_vector,
        )
        n = graph.reservoir_node_count
        nw = graph.well_node_count
        defect = np.concatenate(
            [
                graph.node_features[:n, 26],
                graph.node_features[:n, 27],
                graph.node_features[:n, 28],
                graph.node_features[n : n + nw, 29],
            ]
        )
        return self.relaxation * defect.astype(float)


def candidate_residual_ratio(
    model: ThreePhaseMultiContinuumModel,
    current: State,
    previous: State | None,
    dt_s: float,
    previous_dt_s: float | None,
    alpha: float,
) -> float:
    x_ext = NonlinearSolver.extrapolate(
        model, current, previous, dt_s, previous_dt_s
    )
    correction = FixedDefectPredictor(alpha).predict_correction(
        model, current, previous, dt_s, x_ext
    )
    candidate = model.warm_start_well_projection(x_ext + correction)
    base = np.linalg.norm(model.residual(x_ext, current, dt_s), ord=np.inf)
    trial = np.linalg.norm(model.residual(candidate, current, dt_s), ord=np.inf)
    return float(trial / max(base, 1.0e-30))


def solve_with_alpha(
    model: ThreePhaseMultiContinuumModel,
    current: State,
    previous: State | None,
    dt_s: float,
    previous_dt_s: float | None,
    alpha: float,
) -> SolverResult:
    cfg = model.cfg.solver
    cfg.mode = "S2"
    solver = NonlinearSolver(model, cfg, FixedDefectPredictor(alpha))
    return solver.solve_timestep(current, dt_s, previous, previous_dt_s)


def run_oracle_schedule(
    config_path: Path,
    alphas: tuple[float, ...],
    criterion: str,
) -> tuple[ThreePhaseMultiContinuumModel, list[State], list[TimeStepRecord], list[dict[str, object]]]:
    cfg = load_case(config_path)
    cfg.time.adaptive = False
    cfg.solver.mode = "S2"
    model = ThreePhaseMultiContinuumModel(cfg, build_case_grid(cfg))
    current = model.initial_state()
    previous: State | None = None
    previous_dt_s: float | None = None
    states = [current.copy()]
    records: list[TimeStepRecord] = []
    step_rows: list[dict[str, object]] = []
    time_s = 0.0
    dt_s = cfg.time.initial_dt_s
    step_index = 0

    while time_s < cfg.time.total_time_s - 1.0e-12:
        dt_s = min(cfg.time.initial_dt_s, cfg.time.total_time_s - time_s)
        ratios = {
            alpha: candidate_residual_ratio(
                model, current, previous, dt_s, previous_dt_s, alpha
            )
            for alpha in alphas
        }
        if criterion == "residual":
            selected_alpha = min(alphas, key=lambda alpha: (ratios[alpha], alpha))
            selected = solve_with_alpha(
                model, current, previous, dt_s, previous_dt_s, selected_alpha
            )
            evaluations = {selected_alpha: selected}
        elif criterion == "margin":
            teacher_alphas = tuple(alpha for alpha in alphas if alpha <= 1.0)
            best_alpha = min(
                teacher_alphas, key=lambda alpha: (ratios[alpha], alpha)
            )
            ratio_at_one = ratios.get(1.0, float("inf"))
            if (
                best_alpha <= 0.125
                and ratio_at_one <= ratios[best_alpha] + 0.05
            ):
                selected_alpha = 0.0
            else:
                feasible = [
                    alpha for alpha in teacher_alphas
                    if alpha > 0.0 and ratios[alpha] <= 0.90
                ]
                selected_alpha = (
                    max(feasible)
                    if feasible
                    else best_alpha if ratios[best_alpha] < 1.0 else 0.0
                )
            selected = solve_with_alpha(
                model, current, previous, dt_s, previous_dt_s, selected_alpha
            )
            evaluations = {selected_alpha: selected}
        elif criterion == "newton":
            evaluations = {
                alpha: solve_with_alpha(
                    model, current, previous, dt_s, previous_dt_s, alpha
                )
                for alpha in alphas
            }
            converged = {
                alpha: result
                for alpha, result in evaluations.items()
                if result.converged
            }
            if not converged:
                selected_alpha = 0.0
                selected = evaluations[selected_alpha]
            else:
                selected_alpha, selected = min(
                    converged.items(),
                    key=lambda item: (
                        item[1].iterations,
                        item[1].linear_iterations,
                        ratios[item[0]],
                        abs(item[0] - 0.5),
                    ),
                )
        else:
            raise ValueError(f"unknown oracle criterion: {criterion}")

        record = TimeStepRecord(
            time_s=time_s + dt_s if selected.converged else time_s,
            dt_s=dt_s,
            accepted=selected.converged,
            retries=0,
            result=selected,
        )
        records.append(record)
        step_rows.append(
            {
                "step": step_index,
                "time_s": record.time_s,
                "dt_s": dt_s,
                "criterion": criterion,
                "selected_alpha": selected_alpha,
                "selected_residual_ratio": ratios[selected_alpha],
                "newton_iterations": selected.iterations,
                "linear_iterations": selected.linear_iterations,
                "converged": selected.converged,
                "candidate_newton_iterations": {
                    f"{alpha:g}": result.iterations
                    for alpha, result in evaluations.items()
                },
                "candidate_residual_ratios": {
                    f"{alpha:g}": ratios[alpha] for alpha in alphas
                },
            }
        )
        print(
            f"{criterion} step={step_index:02d} alpha={selected_alpha:.3f} "
            f"ratio={ratios[selected_alpha]:.6e} "
            f"newton={selected.iterations} linear={selected.linear_iterations} "
            f"converged={int(selected.converged)}",
            flush=True,
        )
        if not selected.converged:
            break
        previous, current = current, selected.state
        previous_dt_s = dt_s
        states.append(current.copy())
        time_s += dt_s
        step_index += 1
    return model, states, records, step_rows


def write_steps(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "time_s",
        "dt_s",
        "criterion",
        "selected_alpha",
        "selected_residual_ratio",
        "newton_iterations",
        "linear_iterations",
        "converged",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the 41x41 scalar-relaxation ceiling before changing the "
            "v0.18 model or training data."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/oracle_41x41_v18"))
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
    )
    args = parser.parse_args()
    alphas = tuple(sorted({float(value) for value in args.alphas}))
    if not alphas or alphas[0] < 0.0 or 0.0 not in alphas:
        raise ValueError("alpha scan must contain 0.0 and non-negative values")

    baseline_cfg = load_case(args.config)
    baseline_cfg.time.adaptive = False
    baseline_cfg.solver.mode = "S1"
    baseline_model = ThreePhaseMultiContinuumModel(
        baseline_cfg, build_case_grid(baseline_cfg)
    )
    started = perf_counter()
    baseline_states, baseline_records = run_schedule(
        NonlinearSolver(baseline_model, baseline_cfg.solver),
        baseline_model.initial_state(),
        baseline_cfg.time.total_time_s,
        baseline_cfg.time.initial_dt_s,
        baseline_cfg.time.min_dt_s,
        baseline_cfg.time.max_dt_s,
        adaptive=False,
    )
    baseline_wall_s = perf_counter() - started
    save_run(args.output / "S1", baseline_states, baseline_records, baseline_model)

    results: dict[str, object] = {
        "version": "0.18",
        "purpose": "41x41 scalar physical-defect theoretical ceiling",
        "config": str(args.config),
        "alphas": list(alphas),
        "S1": summarize(baseline_records),
        "S1_diagnostic_wall_s": baseline_wall_s,
    }
    baseline_newton = sum(
        item.result.iterations for item in baseline_records if item.accepted
    )
    baseline_linear = sum(
        item.result.linear_iterations for item in baseline_records if item.accepted
    )

    for criterion in ("residual", "margin", "newton"):
        started = perf_counter()
        model, states, records, rows = run_oracle_schedule(
            args.config, alphas, criterion
        )
        diagnostic_wall_s = perf_counter() - started
        directory = args.output / f"oracle_{criterion}"
        save_run(directory, states, records, model)
        write_steps(directory / "selected_alphas.csv", rows)
        accepted = [item for item in records if item.accepted]
        total_newton = sum(item.result.iterations for item in accepted)
        total_linear = sum(item.result.linear_iterations for item in accepted)
        final_state = states[-1]
        reference = baseline_states[-1]
        summary = summarize(records)
        summary.update(
            {
                "diagnostic_wall_s": diagnostic_wall_s,
                "total_newton_iterations": total_newton,
                "newton_reduction_vs_S1_percent": (
                    100.0 * (1.0 - total_newton / baseline_newton)
                    if baseline_newton > 0
                    else None
                ),
                "linear_reduction_vs_S1_percent": (
                    100.0 * (1.0 - total_linear / baseline_linear)
                    if baseline_linear > 0
                    else None
                ),
                "selected_alpha_median": float(
                    np.median([row["selected_alpha"] for row in rows])
                ),
                "selected_alpha_counts": {
                    f"{alpha:g}": sum(row["selected_alpha"] == alpha for row in rows)
                    for alpha in alphas
                },
                "pressure_relative_l2_vs_S1": float(
                    np.linalg.norm(final_state.pressure_pa - reference.pressure_pa)
                    / max(np.linalg.norm(reference.pressure_pa), 1.0e-30)
                ),
                "water_saturation_mae_vs_S1": float(
                    np.mean(
                        np.abs(
                            final_state.water_saturation
                            - reference.water_saturation
                        )
                    )
                ),
            }
        )
        results[f"oracle_{criterion}"] = summary
        results[f"oracle_{criterion}_step_details"] = rows

    newton_gain = results["oracle_newton"]["newton_reduction_vs_S1_percent"]
    residual_gain = results["oracle_residual"]["newton_reduction_vs_S1_percent"]
    margin_gain = results["oracle_margin"]["newton_reduction_vs_S1_percent"]
    if newton_gain is not None and newton_gain >= 5.0:
        decision = "MULTISCALE_TRAINING_HAS_HEADROOM"
    elif newton_gain is not None and newton_gain > 0.0:
        decision = "LIMITED_HEADROOM_IMPROVE_DIRECTION_FIRST"
    else:
        decision = "NO_SCALAR_HEADROOM_REBUILD_PHYSICS_DIRECTION"
    results["decision"] = decision
    results["teacher_alignment"] = {
        "residual_teacher_newton_reduction_percent": residual_gain,
        "newton_margin_teacher_reduction_percent": margin_gain,
        "newton_oracle_reduction_percent": newton_gain,
        "gap_percent_points": (
            newton_gain - residual_gain
            if newton_gain is not None and residual_gain is not None
            else None
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False)
    (args.output / "oracle_summary.json").write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
