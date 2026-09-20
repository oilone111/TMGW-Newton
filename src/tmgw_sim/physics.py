from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CaseConfig
from .constitutive import (
    boundary_layer_factor,
    mixed_matrix_permeability,
    three_phase_mobility,
    stress_sensitivity_factor,
    threshold_drive,
)
from .grid import REL_FD, REL_FF_DISCRETE, REL_MF, REL_MM, MultiContinuumGrid
from .wells import peaceman_well_index


@dataclass
class State:
    pressure_pa: np.ndarray
    water_saturation: np.ndarray
    gas_saturation: np.ndarray | None = None
    well_pressure_pa: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.gas_saturation is None:
            self.gas_saturation = np.zeros_like(self.water_saturation)

    def copy(self) -> "State":
        return State(
            self.pressure_pa.copy(),
            self.water_saturation.copy(),
            self.gas_saturation.copy(),
            None if self.well_pressure_pa is None else self.well_pressure_pa.copy(),
        )


@dataclass
class FluxResult:
    oil_m3_s: np.ndarray
    water_m3_s: np.ndarray
    gas_m3_s: np.ndarray
    total_m3_s: np.ndarray
    hard_gate: np.ndarray
    transmissibility_m3_pa_s: np.ndarray


class ThreePhaseMultiContinuumModel:
    """Fully implicit oil-water-gas, multi-continuum finite-volume model.

    Pressure, water saturation and gas saturation are independent variables;
    oil saturation follows from phase closure.  The current analytical SCAL
    and constant-viscosity PVT interfaces are replaceable by field tables.
    """

    pressure_scale_pa = 1.0e7

    def __init__(self, cfg: CaseConfig, grid: MultiContinuumGrid):
        self.cfg = cfg
        self.grid = grid
        self.n = grid.n_nodes
        self.n_wells = len(cfg.wells)
        self._porosity = self._build_porosity()
        self._base_perm = self._build_base_permeability()
        self._well_node = self._resolve_well_nodes()

    def initial_state(self) -> State:
        well_pressure = np.asarray([
            well.value if well.control == "bhp" else self.cfg.initial.pressure_pa
            for well in self.cfg.wells
        ], dtype=float)
        return State(
            np.full(self.n, self.cfg.initial.pressure_pa, dtype=float),
            np.full(self.n, self.cfg.initial.water_saturation, dtype=float),
            np.full(self.n, self.cfg.initial.gas_saturation, dtype=float),
            well_pressure,
        )

    def _build_porosity(self) -> np.ndarray:
        medium = self.grid.node_medium
        return np.where(
            medium == 0,
            self.cfg.rock.porosity,
            np.where(medium == 1, self.cfg.fracture.continuum_porosity, self.cfg.fracture.discrete_porosity),
        ).astype(float)

    def _build_base_permeability(self) -> np.ndarray:
        medium = self.grid.node_medium
        return np.where(
            medium == 0,
            self.cfg.rock.permeability_m2,
            np.where(
                medium == 1,
                self.cfg.rock.permeability_m2 * self.cfg.fracture.continuum_multiplier,
                self.cfg.fracture.discrete_permeability_m2,
            ),
        ).astype(float)

    def _resolve_well_nodes(self) -> list[int]:
        discrete_by_cell = {
            int(c): int(n) for c, n in zip(
                self.grid.node_base_cell[self.grid.discrete_global_nodes],
                self.grid.discrete_global_nodes,
            )
        }
        return [discrete_by_cell.get(w.cell, w.cell) for w in self.cfg.wells]

    def pack(self, state: State) -> np.ndarray:
        well_pressure = state.well_pressure_pa
        if well_pressure is None:
            well_pressure = np.asarray([
                well.value if well.control == "bhp" else self.cfg.initial.pressure_pa
                for well in self.cfg.wells
            ], dtype=float)
        if well_pressure.size != self.n_wells:
            raise ValueError(f"well pressure vector must contain {self.n_wells} entries")
        return np.concatenate([
            state.pressure_pa / self.pressure_scale_pa,
            state.water_saturation,
            state.gas_saturation,
            well_pressure / self.pressure_scale_pa,
        ])

    def unpack(self, vector: np.ndarray) -> State:
        expected = 3 * self.n + self.n_wells
        if vector.size != expected:
            raise ValueError(f"state vector must contain {expected} entries")
        return State(
            vector[: self.n] * self.pressure_scale_pa,
            vector[self.n : 2 * self.n].copy(),
            vector[2 * self.n : 3 * self.n].copy(),
            vector[3 * self.n :] * self.pressure_scale_pa,
        )

    def permeability(self, pressure_pa: np.ndarray) -> np.ndarray:
        result = self._base_perm.copy()
        matrix = self.grid.node_medium == 0
        pore_k = result[matrix] * stress_sensitivity_factor(
            pressure_pa[matrix], self.cfg.rock.permeability_m2, self.cfg.microfracture
        )
        result[matrix] = mixed_matrix_permeability(
            pore_k, self.cfg.rock.porosity, 1.0, self.cfg.microfracture
        )
        return np.maximum(result, 1e-24)

    def edge_fluxes(self, state: State) -> FluxResult:
        g = self.grid
        i, j = g.edge_i, g.edge_j
        dp = state.pressure_pa[i] - state.pressure_pa[j]
        perm = self._base_perm.copy()
        matrix_node = g.node_medium == 0
        perm[matrix_node] *= stress_sensitivity_factor(
            state.pressure_pa[matrix_node], self.cfg.rock.permeability_m2,
            self.cfg.microfracture,
        )
        harmonic_k = 2.0 * perm[i] * perm[j] / np.maximum(perm[i] + perm[j], 1e-40)

        rel = g.edge_relation
        matrix_edge = rel == REL_MM
        threshold_edge = np.isin(rel, (REL_MM, REL_MF))
        boundary = np.ones(g.n_edges)
        gradient = np.abs(dp) / np.maximum(g.edge_length, 1e-20)
        boundary[matrix_edge] = boundary_layer_factor(
            gradient[matrix_edge], self.cfg.microfracture
        )
        harmonic_k[matrix_edge] = mixed_matrix_permeability(
            harmonic_k[matrix_edge], self.cfg.rock.porosity,
            boundary[matrix_edge], self.cfg.microfracture,
        )
        geom = harmonic_k * g.edge_area / np.maximum(g.edge_length, 1e-20)
        geom = geom.copy()
        geom[rel == REL_MF] *= self.cfg.fracture.continuum_transfer_shape * self.cfg.fracture.mf_transfer_multiplier
        geom[rel == REL_FD] *= self.cfg.fracture.fF_transfer_multiplier

        hard_gate = np.ones(g.n_edges, dtype=float)
        drive = dp.copy()
        if self.cfg.microfracture.enabled:
            thresholded = threshold_drive(
                dp[threshold_edge],
                g.edge_length[threshold_edge],
                self.cfg.microfracture.threshold_gradient_pa_m,
            )
            hard_gate[threshold_edge] = (np.abs(thresholded) > 0.0).astype(float)
            drive[threshold_edge] = thresholded
            geom[matrix_edge] *= boundary[matrix_edge]

        mob_o, mob_w, mob_g = three_phase_mobility(
            state.water_saturation, state.gas_saturation, self.cfg.fluid
        )
        upstream = np.where(drive >= 0.0, i, j)
        qo = geom * mob_o[upstream] * drive
        qw = geom * mob_w[upstream] * drive
        qg = geom * mob_g[upstream] * drive

        # Forchheimer correction on discrete-main-fracture internal edges.
        discrete_edge = rel == REL_FF_DISCRETE
        if np.any(discrete_edge):
            total_darcy = qo[discrete_edge] + qw[discrete_edge] + qg[discrete_edge]
            area = np.maximum(g.edge_area[discrete_edge], 1e-20)
            inertial = self.cfg.fracture.forchheimer_beta_m_inv * g.edge_length[discrete_edge] / area
            magnitude = np.abs(total_darcy)
            correction = np.where(
                magnitude > 0.0,
                2.0 / (1.0 + np.sqrt(1.0 + 4.0 * inertial * magnitude)),
                1.0,
            )
            qo[discrete_edge] *= correction
            qw[discrete_edge] *= correction
            qg[discrete_edge] *= correction

        total = qo + qw + qg
        teff = np.divide(total, drive, out=np.zeros_like(total), where=np.abs(drive) > 0.0)
        return FluxResult(qo, qw, qg, total, hard_gate, teff)

    def well_sources(
        self, state: State
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        q_o = np.zeros(self.n)
        q_w = np.zeros(self.n)
        q_g = np.zeros(self.n)
        diagnostics: dict[str, float] = {}
        mob_o, mob_w, mob_g = three_phase_mobility(
            state.water_saturation, state.gas_saturation, self.cfg.fluid
        )
        g = self.cfg.grid
        permeability = self.permeability(state.pressure_pa)

        for well_index, (well, node) in enumerate(zip(self.cfg.wells, self._well_node)):
            if well.control == "rate":
                total = well.value
                if total >= 0.0:
                    qo, qw, qg = 0.0, total, 0.0
                else:
                    total_mob = mob_o[node] + mob_w[node] + mob_g[node]
                    qo = total * mob_o[node] / max(total_mob, 1e-30)
                    qw = total * mob_w[node] / max(total_mob, 1e-30)
                    qg = total * mob_g[node] / max(total_mob, 1e-30)
            elif well.control == "bhp":
                aperture = self.cfg.fracture.discrete_aperture_m if self.grid.node_medium[node] == 2 else None
                wi, re = peaceman_well_index(
                    permeability[node], g.dz_m, g.dx_m, g.dy_m, well.radius_m, well.skin, aperture
                )
                bhp = well.value if state.well_pressure_pa is None else state.well_pressure_pa[well_index]
                draw = bhp - state.pressure_pa[node]
                total_mob = mob_o[node] + mob_w[node] + mob_g[node]
                total = wi * total_mob * draw
                if total >= 0.0:
                    qo, qw, qg = 0.0, total, 0.0
                else:
                    qo = total * mob_o[node] / max(total_mob, 1e-30)
                    qw = total * mob_w[node] / max(total_mob, 1e-30)
                    qg = total * mob_g[node] / max(total_mob, 1e-30)
                diagnostics[f"{well.name}.re_m"] = re
            else:
                raise ValueError(f"unsupported well control: {well.control}")
            q_o[node] += qo
            q_w[node] += qw
            q_g[node] += qg
            diagnostics[f"{well.name}.q_total_m3_s"] = qo + qw + qg
            diagnostics[f"{well.name}.q_water_m3_s"] = qw
            diagnostics[f"{well.name}.q_gas_m3_s"] = qg
        return q_o, q_w, q_g, diagnostics

    def well_control_residual(self, state: State) -> np.ndarray:
        if self.n_wells == 0:
            return np.empty(0, dtype=float)
        if state.well_pressure_pa is None:
            raise ValueError("well pressures are required by the coupled state")
        residual = np.zeros(self.n_wells, dtype=float)
        mob_o, mob_w, mob_g = three_phase_mobility(
            state.water_saturation, state.gas_saturation, self.cfg.fluid
        )
        permeability = self.permeability(state.pressure_pa)
        g = self.cfg.grid
        for well_index, (well, node) in enumerate(zip(self.cfg.wells, self._well_node)):
            if well.control == "bhp":
                target = well.value
            else:
                aperture = (
                    self.cfg.fracture.discrete_aperture_m
                    if self.grid.node_medium[node] == 2 else None
                )
                wi, _ = peaceman_well_index(
                    permeability[node], g.dz_m, g.dx_m, g.dy_m,
                    well.radius_m, well.skin, aperture,
                )
                total_mob = mob_o[node] + mob_w[node] + mob_g[node]
                target = state.pressure_pa[node] + well.value / max(wi * total_mob, 1e-30)
            residual[well_index] = (
                state.well_pressure_pa[well_index] - target
            ) / self.pressure_scale_pa
        return residual

    def residual(self, vector: np.ndarray, old: State, dt_s: float) -> np.ndarray:
        state = self.unpack(vector)
        flux = self.edge_fluxes(state)
        q_o_well, q_w_well, q_g_well, _ = self.well_sources(state)
        divergence_total = np.zeros(self.n)
        divergence_water = np.zeros(self.n)
        divergence_gas = np.zeros(self.n)
        np.add.at(divergence_total, self.grid.edge_i, flux.total_m3_s)
        np.add.at(divergence_total, self.grid.edge_j, -flux.total_m3_s)
        np.add.at(divergence_water, self.grid.edge_i, flux.water_m3_s)
        np.add.at(divergence_water, self.grid.edge_j, -flux.water_m3_s)
        np.add.at(divergence_gas, self.grid.edge_i, flux.gas_m3_s)
        np.add.at(divergence_gas, self.grid.edge_j, -flux.gas_m3_s)

        pore_volume = np.maximum(self._porosity * self.grid.node_volume, 1e-20)
        ct = self.cfg.rock.compressibility_pa_inv
        pressure_balance = (
            ct * (state.pressure_pa - old.pressure_pa)
            + dt_s * (divergence_total - q_o_well - q_w_well - q_g_well) / pore_volume
        )
        water_balance = (
            state.water_saturation - old.water_saturation
            + dt_s * (divergence_water - q_w_well) / pore_volume
        )
        gas_balance = (
            state.gas_saturation - old.gas_saturation
            + dt_s * (divergence_gas - q_g_well) / pore_volume
        )
        return np.concatenate([
            pressure_balance,
            water_balance,
            gas_balance,
            self.well_control_residual(state),
        ])

    def physical_projection(self, vector: np.ndarray) -> np.ndarray:
        projected = vector.copy()
        projected[: self.n] = np.clip(projected[: self.n], 0.1, 20.0)
        sw_min = self.cfg.fluid.swc
        sg_min = self.cfg.fluid.sgc
        sw = np.maximum(projected[self.n : 2 * self.n], sw_min)
        sg = np.maximum(projected[2 * self.n : 3 * self.n], sg_min)
        mobile_limit = max(1.0 - self.cfg.fluid.sor - sw_min - sg_min, 0.0)
        excess_w = sw - sw_min
        excess_g = sg - sg_min
        excess_total = excess_w + excess_g
        scale = np.minimum(1.0, mobile_limit / np.maximum(excess_total, 1e-30))
        projected[self.n : 2 * self.n] = sw_min + scale * excess_w
        projected[2 * self.n : 3 * self.n] = sg_min + scale * excess_g
        if self.n_wells:
            projected[3 * self.n :] = np.clip(projected[3 * self.n :], 0.1, 20.0)
        return projected

    def warm_start_well_projection(self, vector: np.ndarray) -> np.ndarray:
        """Enforce algebraic well controls on a proposed warm-start state.

        This projection is used only before Newton.  BHP-controlled wells are
        fixed exactly, while rate-controlled well pressure is recovered from
        the Peaceman relation at the proposed reservoir state.
        """
        projected = self.physical_projection(vector)
        if not self.n_wells:
            return projected
        state = self.unpack(projected)
        mob_o, mob_w, mob_g = three_phase_mobility(
            state.water_saturation, state.gas_saturation, self.cfg.fluid
        )
        permeability_now = self.permeability(state.pressure_pa)
        for well_index, (well, node) in enumerate(zip(self.cfg.wells, self._well_node)):
            if well.control == "bhp":
                state.well_pressure_pa[well_index] = well.value
                continue
            aperture = (
                self.cfg.fracture.discrete_aperture_m
                if self.grid.node_medium[node] == 2 else None
            )
            wi, _ = peaceman_well_index(
                permeability_now[node], self.cfg.grid.dz_m,
                self.cfg.grid.dx_m, self.cfg.grid.dy_m,
                well.radius_m, well.skin, aperture,
            )
            total_mob = mob_o[node] + mob_w[node] + mob_g[node]
            state.well_pressure_pa[well_index] = (
                state.pressure_pa[node] + well.value / max(wi * total_mob, 1.0e-30)
            )
        return self.physical_projection(self.pack(state))

    def mass_imbalance(self, state: State, old: State, dt_s: float) -> float:
        residual = self.residual(self.pack(state), old, dt_s)
        return float(np.max(np.abs(residual)))

    def global_mass_error(self, state: State, old: State, dt_s: float) -> float:
        flux = self.edge_fluxes(state)
        q_o_well, q_w_well, q_g_well, _ = self.well_sources(state)
        pore_volume = np.maximum(self._porosity * self.grid.node_volume, 1e-20)
        accumulation = (
            self.cfg.rock.compressibility_pa_inv
            * pore_volume
            * (state.pressure_pa - old.pressure_pa)
            / max(dt_s, 1e-30)
        )
        source = q_o_well + q_w_well + q_g_well
        # Internal edge fluxes cancel exactly in the global balance.
        imbalance = abs(float(np.sum(accumulation - source)))
        throughput = max(float(np.sum(np.abs(source))), float(np.sum(np.abs(flux.total_m3_s))), 1e-30)
        total_error = imbalance / throughput
        water_rate = pore_volume * (state.water_saturation - old.water_saturation) / max(dt_s, 1e-30)
        gas_rate = pore_volume * (state.gas_saturation - old.gas_saturation) / max(dt_s, 1e-30)
        water_error = abs(float(np.sum(water_rate - q_w_well))) / max(
            float(np.sum(np.abs(q_w_well))), float(np.sum(np.abs(flux.water_m3_s))), 1e-30
        )
        gas_error = abs(float(np.sum(gas_rate - q_g_well))) / max(
            float(np.sum(np.abs(q_g_well))), float(np.sum(np.abs(flux.gas_m3_s))), 1e-30
        )
        return max(total_error, water_error, gas_error)


# Compatibility alias for scripts created before the three-phase upgrade.
TwoPhaseMultiContinuumModel = ThreePhaseMultiContinuumModel
