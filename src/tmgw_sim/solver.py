from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Protocol

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .config import SolverConfig
from .physics import State, ThreePhaseMultiContinuumModel


class CorrectionPredictor(Protocol):
    def predict_correction(
        self,
        model: ThreePhaseMultiContinuumModel,
        current: State,
        previous: State | None,
        dt_s: float,
        extrapolated_vector: np.ndarray,
    ) -> np.ndarray: ...


@dataclass
class IterationRecord:
    iteration: int
    residual_norm: float
    increment_norm: float
    damping: float
    linear_iterations: int


@dataclass
class SolverResult:
    state: State
    converged: bool
    iterations: int
    initial_residual: float
    final_residual: float
    linear_iterations: int
    elapsed_s: float
    initial_source: str
    residual_ratio: float | None = None
    warm_start_mass_ratio: float | None = None
    warm_start_alpha: float | None = None
    predictor_elapsed_s: float = 0.0
    final_mass_error: float = float("nan")
    records: list[IterationRecord] = field(default_factory=list)


class NonlinearSolver:
    """Damped Jacobian-free Newton-Krylov solver with S0--S3 warm starts."""

    def __init__(
        self,
        model: ThreePhaseMultiContinuumModel,
        cfg: SolverConfig,
        predictor: CorrectionPredictor | None = None,
    ):
        self.model = model
        self.cfg = cfg
        self.predictor = predictor
        self._precomputed_initial_residual: np.ndarray | None = None

    @staticmethod
    def extrapolate(
        model: ThreePhaseMultiContinuumModel,
        current: State,
        previous: State | None,
        dt_s: float,
        previous_dt_s: float | None,
    ) -> np.ndarray:
        x_n = model.pack(current)
        if previous is None or previous_dt_s is None or previous_dt_s <= 0.0:
            return x_n.copy()
        theta = min(dt_s / previous_dt_s, 1.0)
        return model.physical_projection(x_n + theta * (x_n - model.pack(previous)))

    def _select_initial(
        self,
        current: State,
        previous: State | None,
        dt_s: float,
        previous_dt_s: float | None,
    ) -> tuple[np.ndarray, str, float | None, float | None, float | None, float]:
        self._precomputed_initial_residual = None
        x_previous = self.model.pack(current)
        x_ext = self.extrapolate(self.model, current, previous, dt_s, previous_dt_s)
        mode = self.cfg.mode.upper()
        if mode == "S0":
            return x_previous, "previous_state", None, None, None, 0.0
        if mode == "S1":
            return x_ext, "temporal_extrapolation", None, None, None, 0.0
        if mode not in {"S2", "S3"}:
            raise ValueError(f"unknown solver mode {self.cfg.mode}")
        if self.predictor is None:
            raise RuntimeError(f"solver mode {mode} requires a correction predictor")

        # S3 needs the extrapolated residual for both its safety gate and the
        # first Newton check.  Compute it once, reuse it in graph construction,
        # and bypass graph inference away from the first step and the narrow
        # iteration-boundary region.  S2 remains an intentionally unprotected
        # learned baseline and therefore always invokes its predictor.
        r_ext_vector: np.ndarray | None = None
        r_ext: float | None = None
        mass_ext: float | None = None
        if mode == "S3":
            r_ext_vector = self.model.residual(x_ext, current, dt_s)
            r_ext = float(np.linalg.norm(r_ext_vector, ord=np.inf))
            trigger = float(getattr(self.cfg, "warm_start_trigger_residual", 0.0))
            if previous is not None and trigger > 0.0 and r_ext > trigger:
                self._precomputed_initial_residual = r_ext_vector
                return x_ext, "triggered_s1", 1.0, 1.0, 0.0, 0.0
            mass_ext = self.model.global_mass_error(
                self.model.unpack(x_ext), current, dt_s
            )
            try:
                setattr(self.predictor, "precomputed_residual", r_ext_vector)
            except (AttributeError, TypeError):
                pass

        predictor_started = perf_counter()
        correction = self.predictor.predict_correction(
            self.model, current, previous, dt_s, x_ext
        )
        predicted_relaxation = float(
            getattr(self.predictor, "last_relaxation", 1.0)
        )
        predictor_elapsed = perf_counter() - predictor_started
        if correction.shape != x_ext.shape:
            raise ValueError("predictor returned a correction with an invalid shape")
        x_ai = self.model.warm_start_well_projection(x_ext + correction)
        if mode == "S2":
            return (
                x_ai,
                "graph_warm_start",
                None,
                None,
                predicted_relaxation,
                predictor_elapsed,
            )

        assert r_ext_vector is not None and r_ext is not None and mass_ext is not None
        if not np.all(np.isfinite(correction)):
            self._precomputed_initial_residual = r_ext_vector
            return x_ext, "fallback_nonfinite", None, None, 0.0, predictor_elapsed

        # v0.18 predicts one compact graph-level relaxation and verifies it
        # once. P0 uses the identical alpha=(1,) contract, so any improvement
        # over P0 comes from the learned coefficient rather than a wider
        # candidate search. The exact S1 state remains the safe fallback.
        candidates: list[
            tuple[float, float, float, np.ndarray, np.ndarray]
        ] = []
        candidate_alphas = getattr(
            self.predictor, "candidate_alphas",
            (2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625),
        )
        for alpha in candidate_alphas:
            blended = self.model.warm_start_well_projection(x_ext + alpha * correction)
            if not np.all(np.isfinite(blended)):
                continue
            blended_residual_vector = self.model.residual(blended, current, dt_s)
            blended_residual = np.linalg.norm(blended_residual_vector, ord=np.inf)
            blended_ratio = blended_residual / max(r_ext, 1e-30)
            blended_mass = self.model.global_mass_error(
                self.model.unpack(blended), current, dt_s
            )
            blended_mass_ratio = blended_mass / max(
                mass_ext, self.cfg.warm_start_mass_tolerance
            )
            if blended_mass_ratio <= self.cfg.residual_reject:
                candidates.append((
                    blended_ratio, blended_mass_ratio, alpha, blended,
                    blended_residual_vector,
                ))
        if candidates:
            ratio_best, mass_ratio_best, alpha_best, state_best, residual_best = min(
                candidates, key=lambda item: item[0]
            )
            if ratio_best <= self.cfg.residual_use_max:
                # A residual curve that is almost unchanged between the half
                # and full learned step is a projection-saturated response.
                # The v0.17 41x41 second step had exactly this signature:
                # residual decreased, yet Newton increased from five to seven.
                probe_min = float(getattr(
                    self.cfg, "warm_start_flat_probe_min_relaxation", 0.35
                ))
                probe_max = float(getattr(
                    self.cfg, "warm_start_flat_probe_max_relaxation", 0.75
                ))
                probe_tolerance = float(getattr(
                    self.cfg, "warm_start_flat_ratio_tolerance", 0.05
                ))
                if probe_min <= predicted_relaxation <= probe_max:
                    half_state = self.model.warm_start_well_projection(
                        x_ext + 0.5 * alpha_best * correction
                    )
                    half_residual = float(np.linalg.norm(
                        self.model.residual(half_state, current, dt_s), ord=np.inf
                    ))
                    half_ratio = half_residual / max(r_ext, 1e-30)
                    if abs(half_ratio - ratio_best) <= probe_tolerance:
                        self._precomputed_initial_residual = r_ext_vector
                        return (
                            x_ext, "fallback_flat_response", ratio_best,
                            mass_ratio_best, 0.0, predictor_elapsed,
                        )
                self._precomputed_initial_residual = residual_best
                return (
                    state_best,
                    "accepted_ai" if ratio_best <= self.cfg.residual_accept else "residual_blend",
                    ratio_best, mass_ratio_best,
                    predicted_relaxation * alpha_best, predictor_elapsed,
                )
        best_ratio = min((item[0] for item in candidates), default=None)
        best_mass = min(candidates, key=lambda item: item[0])[1] if candidates else None
        self._precomputed_initial_residual = r_ext_vector
        return x_ext, "fallback_residual", best_ratio, best_mass, 0.0, predictor_elapsed

    def solve_timestep(
        self,
        current: State,
        dt_s: float,
        previous: State | None = None,
        previous_dt_s: float | None = None,
    ) -> SolverResult:
        started = perf_counter()
        x, source, ratio, mass_ratio, warm_start_alpha, predictor_elapsed = self._select_initial(
            current, previous, dt_s, previous_dt_s
        )
        residual_fn = lambda z: self.model.residual(z, current, dt_s)
        r = (
            self._precomputed_initial_residual
            if self._precomputed_initial_residual is not None
            else residual_fn(x)
        )
        self._precomputed_initial_residual = None
        initial_norm = float(np.linalg.norm(r, ord=np.inf))
        initial_mass_error = self.model.global_mass_error(self.model.unpack(x), current, dt_s)
        total_linear = 0
        records: list[IterationRecord] = []

        if initial_norm < self.cfg.nonlinear_tolerance and initial_mass_error < self.cfg.mass_tolerance:
            return SolverResult(
                state=self.model.unpack(x), converged=True, iterations=0,
                initial_residual=initial_norm, final_residual=initial_norm,
                linear_iterations=0, elapsed_s=perf_counter() - started,
                initial_source=source, residual_ratio=ratio,
                warm_start_mass_ratio=mass_ratio,
                warm_start_alpha=warm_start_alpha,
                predictor_elapsed_s=predictor_elapsed,
                final_mass_error=initial_mass_error, records=records,
            )

        converged = False
        increment_norm = np.inf
        for iteration in range(1, self.cfg.max_newton + 1):
            r = residual_fn(x)
            r_norm = float(np.linalg.norm(r, ord=np.inf))
            current_mass_error = self.model.global_mass_error(self.model.unpack(x), current, dt_s)
            if (
                r_norm < self.cfg.nonlinear_tolerance
                and increment_norm < self.cfg.increment_tolerance
                and current_mass_error < self.cfg.mass_tolerance
            ):
                converged = True
                break

            linear_counter = [0]
            x_norm = max(float(np.linalg.norm(x, ord=np.inf)), 1.0)
            if x.size <= self.cfg.dense_jacobian_limit:
                jacobian = np.empty((x.size, x.size), dtype=float)
                for column in range(x.size):
                    epsilon = np.sqrt(np.finfo(float).eps) * max(abs(x[column]), 1.0)
                    trial_column = x.copy()
                    trial_column[column] += epsilon
                    jacobian[:, column] = (residual_fn(trial_column) - r) / epsilon
                try:
                    delta = np.linalg.solve(jacobian, -r)
                    info = 0
                except np.linalg.LinAlgError:
                    delta, *_ = np.linalg.lstsq(jacobian, -r, rcond=1e-12)
                    info = 0
                linear_counter[0] = 1
            else:
                epsilon_base = np.sqrt(np.finfo(float).eps) * x_norm

                def matvec(direction: np.ndarray) -> np.ndarray:
                    d_norm = max(float(np.linalg.norm(direction)), 1e-30)
                    epsilon = epsilon_base / d_norm
                    return (residual_fn(x + epsilon * direction) - r) / epsilon

                operator = LinearOperator((x.size, x.size), matvec=matvec, dtype=float)

                # Physics-based diagonal preconditioner: accumulation plus the
                # absolute sum of local transmissibilities in each control volume.
                trial_state = self.model.unpack(x)
                edge_flux = self.model.edge_fluxes(trial_state)
                row_conductance = np.zeros(self.model.n)
                np.add.at(row_conductance, self.model.grid.edge_i, np.abs(edge_flux.transmissibility_m3_pa_s))
                np.add.at(row_conductance, self.model.grid.edge_j, np.abs(edge_flux.transmissibility_m3_pa_s))
                pore_volume = np.maximum(
                    self.model._porosity * self.model.grid.node_volume, 1e-20
                )
                diagonal_p = (
                    self.model.cfg.rock.compressibility_pa_inv * self.model.pressure_scale_pa
                    + dt_s * row_conductance * self.model.pressure_scale_pa / pore_volume
                )
                diagonal_sw = np.ones(self.model.n)
                diagonal_sg = np.ones(self.model.n)
                diagonal_well = np.ones(self.model.n_wells)
                diagonal = np.maximum(
                    np.concatenate([diagonal_p, diagonal_sw, diagonal_sg, diagonal_well]),
                    1e-12,
                )
                preconditioner = LinearOperator(
                    (x.size, x.size), matvec=lambda value: value / diagonal, dtype=float
                )

                def callback(_: float) -> None:
                    linear_counter[0] += 1

                delta, info = gmres(
                    operator,
                    -r,
                    M=preconditioner,
                    rtol=self.cfg.gmres_tolerance,
                    atol=0.0,
                    restart=min(30, self.cfg.gmres_maxiter),
                    maxiter=self.cfg.gmres_maxiter,
                    callback=callback,
                    callback_type="legacy",
                )
            total_linear += linear_counter[0]
            if info < 0 or not np.all(np.isfinite(delta)):
                break

            accepted = False
            chosen_damping = 0.0
            trial = x
            for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = self.model.physical_projection(x + damping * delta)
                trial_norm = float(np.linalg.norm(residual_fn(candidate), ord=np.inf))
                if trial_norm <= (1.0 - self.cfg.armijo_c * damping) * r_norm:
                    trial = candidate
                    chosen_damping = damping
                    accepted = True
                    break
            if not accepted:
                break

            increment_norm = float(np.linalg.norm(trial - x, ord=np.inf) / x_norm)
            x = trial
            new_norm = float(np.linalg.norm(residual_fn(x), ord=np.inf))
            records.append(
                IterationRecord(iteration, new_norm, increment_norm, chosen_damping, linear_counter[0])
            )

        final_r = residual_fn(x)
        final_norm = float(np.linalg.norm(final_r, ord=np.inf))
        final_mass_error = self.model.global_mass_error(self.model.unpack(x), current, dt_s)
        if final_norm < self.cfg.nonlinear_tolerance and final_mass_error < self.cfg.mass_tolerance:
            converged = True
        return SolverResult(
            state=self.model.unpack(x), converged=converged, iterations=len(records),
            initial_residual=initial_norm, final_residual=final_norm,
            linear_iterations=total_linear, elapsed_s=perf_counter() - started,
            initial_source=source, residual_ratio=ratio,
            warm_start_mass_ratio=mass_ratio,
            warm_start_alpha=warm_start_alpha,
            predictor_elapsed_s=predictor_elapsed,
            final_mass_error=final_mass_error, records=records,
        )


@dataclass
class TimeStepRecord:
    time_s: float
    dt_s: float
    accepted: bool
    retries: int
    result: SolverResult


def run_schedule(
    solver: NonlinearSolver,
    initial: State,
    total_time_s: float,
    initial_dt_s: float,
    min_dt_s: float,
    max_dt_s: float,
    adaptive: bool = True,
    progress: Callable[[TimeStepRecord], None] | None = None,
) -> tuple[list[State], list[TimeStepRecord]]:
    states = [initial.copy()]
    records: list[TimeStepRecord] = []
    current = initial.copy()
    previous: State | None = None
    previous_dt: float | None = None
    t = 0.0
    dt = initial_dt_s
    fast_steps = 0

    while t < total_time_s - 1e-12:
        dt = min(dt, total_time_s - t)
        retries = 0
        while True:
            result = solver.solve_timestep(current, dt, previous, previous_dt)
            if result.converged:
                step_record = TimeStepRecord(t + dt, dt, True, retries, result)
                records.append(step_record)
                if progress is not None:
                    progress(step_record)
                previous, current = current, result.state
                previous_dt = dt
                states.append(current.copy())
                t += dt
                break
            retries += 1
            dt *= 0.5
            if dt < min_dt_s:
                failed_record = TimeStepRecord(t, dt, False, retries, result)
                records.append(failed_record)
                if progress is not None:
                    progress(failed_record)
                return states, records

        if not adaptive:
            dt = min(initial_dt_s, max_dt_s)
            continue
        if result.iterations <= 4:
            fast_steps += 1
            if fast_steps >= 3:
                dt = min(1.25 * dt, max_dt_s)
                fast_steps = 0
        elif result.iterations >= 10 or any(r.damping < 1.0 for r in result.records):
            dt = max(0.5 * dt, min_dt_s)
            fast_steps = 0
        else:
            fast_steps = 0
    return states, records
