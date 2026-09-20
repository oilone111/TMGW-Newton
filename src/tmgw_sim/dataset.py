from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .graph import GraphArrays, build_graph_arrays
from .physics import State, ThreePhaseMultiContinuumModel
from .solver import NonlinearSolver


@dataclass
class TrainingSample:
    graph: GraphArrays
    target_correction: np.ndarray
    target_state: np.ndarray
    extrapolated_state: np.ndarray
    dt_s: float
    model: ThreePhaseMultiContinuumModel | None = None
    current_state: State | None = None
    baseline_newton_iterations: int = 0
    baseline_initial_residual: float = 0.0
    target_relaxation: float = 0.5
    target_relaxation_residual_ratio: float = 1.0


def calibrate_physics_relaxation(
    model: ThreePhaseMultiContinuumModel,
    current: State,
    dt_s: float,
    extrapolated_state: np.ndarray,
    graph: GraphArrays,
) -> tuple[float, float]:
    """Return the complete-residual teacher for the physics defect direction.

    v0.14 showed that a fixed half step is robust but the best relaxation
    varies systematically between time steps.  v0.16 uses this supervision
    target with the same discrete three-phase residual used by Newton, rather
    than with the converged-state distance that previously degraded the true
    residual during training.
    """
    n = graph.reservoir_node_count
    nw = graph.well_node_count
    defect = np.concatenate([
        graph.node_features[:n, 26],
        graph.node_features[:n, 27],
        graph.node_features[:n, 28],
        graph.node_features[n : n + nw, 29],
    ]).astype(float)
    base_residual = np.linalg.norm(
        model.residual(extrapolated_state, current, dt_s), ord=np.inf
    )
    base_mass = model.global_mass_error(model.unpack(extrapolated_state), current, dt_s)
    candidates: list[tuple[float, float]] = [(1.0, 0.0)]
    for relaxation in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
        candidate = model.warm_start_well_projection(
            extrapolated_state + relaxation * defect
        )
        if not np.all(np.isfinite(candidate)):
            continue
        residual_ratio = float(
            np.linalg.norm(model.residual(candidate, current, dt_s), ord=np.inf)
            / max(base_residual, 1.0e-30)
        )
        mass = model.global_mass_error(model.unpack(candidate), current, dt_s)
        mass_ratio = float(mass / max(
            base_mass, model.cfg.solver.warm_start_mass_tolerance
        ))
        if mass_ratio <= model.cfg.solver.residual_reject:
            candidates.append((residual_ratio, float(relaxation)))
    ratio, relaxation = min(candidates, key=lambda item: item[0])
    return relaxation, ratio


def calibrate_newton_margin_relaxation(
    model: ThreePhaseMultiContinuumModel,
    current: State,
    dt_s: float,
    extrapolated_state: np.ndarray,
    graph: GraphArrays,
    residual_envelope: float = 0.90,
) -> tuple[float, float]:
    """Return the v0.18 Newton-margin teacher.

    The v0.16 teacher minimized the initial residual.  The 41x41 oracle shows
    that this can leave the iterate on the wrong side of a Newton stopping
    boundary: the residual-minimum schedule used 51 Newton steps, while a
    larger but still contractive scalar used 47.  v0.18 therefore chooses the
    largest coherent defect step that remains inside a conservative residual
    envelope.  A nearly flat response at the smallest non-zero scalar is
    treated as an uninformative direction and falls back to S1 (alpha=0).
    """
    n = graph.reservoir_node_count
    nw = graph.well_node_count
    defect = np.concatenate([
        graph.node_features[:n, 26],
        graph.node_features[:n, 27],
        graph.node_features[:n, 28],
        graph.node_features[n : n + nw, 29],
    ]).astype(float)
    base_residual = np.linalg.norm(
        model.residual(extrapolated_state, current, dt_s), ord=np.inf
    )
    base_mass = model.global_mass_error(model.unpack(extrapolated_state), current, dt_s)
    evaluations: list[tuple[float, float]] = [(0.0, 1.0)]
    for relaxation in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
        candidate = model.warm_start_well_projection(
            extrapolated_state + relaxation * defect
        )
        if not np.all(np.isfinite(candidate)):
            continue
        residual_ratio = float(
            np.linalg.norm(model.residual(candidate, current, dt_s), ord=np.inf)
            / max(base_residual, 1.0e-30)
        )
        mass = model.global_mass_error(model.unpack(candidate), current, dt_s)
        mass_ratio = float(mass / max(
            base_mass, model.cfg.solver.warm_start_mass_tolerance
        ))
        if mass_ratio <= model.cfg.solver.residual_reject:
            evaluations.append((float(relaxation), residual_ratio))

    best_alpha, best_ratio = min(evaluations, key=lambda item: (item[1], item[0]))
    ratio_at_one = next(
        (ratio for alpha, ratio in evaluations if abs(alpha - 1.0) < 1.0e-12),
        float("inf"),
    )
    # On the 41x41 second step the residual curve is almost flat from 0.125
    # to 1.0, yet every non-zero defect step costs more Newton work than S1.
    # This guard is scale-free and prevents the same ambiguous response from
    # becoming an aggressive training label in other cases.
    if best_alpha <= 0.125 and ratio_at_one <= best_ratio + 0.05:
        return 0.0, 1.0

    feasible = [
        (alpha, ratio) for alpha, ratio in evaluations
        if alpha > 0.0 and ratio <= residual_envelope
    ]
    if feasible:
        return max(feasible, key=lambda item: item[0])
    if best_ratio < 1.0:
        return best_alpha, best_ratio
    return 0.0, 1.0


def load_run_samples(
    run_dir: str | Path,
    model: ThreePhaseMultiContinuumModel,
    step_indices: list[int] | tuple[int, ...] | np.ndarray | None = None,
    teacher_mode: str = "newton_margin",
) -> list[TrainingSample]:
    run = Path(run_dir)
    arrays = np.load(run / "states.npz")
    pressure = arrays["pressure_pa"]
    saturation = arrays["water_saturation"]
    gas_saturation = arrays["gas_saturation"]
    well_pressure = arrays["well_pressure_pa"]
    with (run / "solver_log.csv").open("r", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("accepted", "true").lower() == "true"
        ]
    if len(rows) < pressure.shape[0] - 1:
        raise RuntimeError(f"{run}: accepted solver log is shorter than the state trajectory")
    samples: list[TrainingSample] = []
    available_steps = pressure.shape[0] - 1
    selected_steps = (
        list(range(available_steps))
        if step_indices is None
        else sorted({int(step) for step in step_indices})
    )
    if any(step < 0 or step >= available_steps for step in selected_steps):
        raise IndexError(f"{run}: requested training step is outside the trajectory")
    for step in selected_steps:
        current = State(
            pressure[step], saturation[step], gas_saturation[step], well_pressure[step]
        )
        previous = None if step == 0 else State(
            pressure[step - 1], saturation[step - 1],
            gas_saturation[step - 1], well_pressure[step - 1],
        )
        target = State(
            pressure[step + 1], saturation[step + 1],
            gas_saturation[step + 1], well_pressure[step + 1],
        )
        dt_s = float(rows[step]["dt_s"])
        previous_dt = None if step == 0 else float(rows[step - 1]["dt_s"])
        extrapolated = NonlinearSolver.extrapolate(model, current, previous, dt_s, previous_dt)
        target_vector = model.pack(target)
        graph = build_graph_arrays(
                model, current, previous, dt_s,
                extrapolated_vector=extrapolated,
            )
        if teacher_mode == "newton_margin":
            target_relaxation, target_relaxation_ratio = calibrate_newton_margin_relaxation(
                model, current, dt_s, extrapolated, graph
            )
        elif teacher_mode == "residual_minimum":
            target_relaxation, target_relaxation_ratio = calibrate_physics_relaxation(
                model, current, dt_s, extrapolated, graph
            )
        else:
            raise ValueError(f"unknown relaxation teacher mode: {teacher_mode}")
        samples.append(TrainingSample(
            graph=graph,
            target_correction=target_vector - extrapolated,
            target_state=target_vector,
            extrapolated_state=extrapolated,
            dt_s=dt_s,
            model=model,
            current_state=current,
            baseline_newton_iterations=int(rows[step]["newton_iterations"]),
            baseline_initial_residual=float(rows[step]["initial_residual"]),
            target_relaxation=target_relaxation,
            target_relaxation_residual_ratio=target_relaxation_ratio,
        ))
    return samples
