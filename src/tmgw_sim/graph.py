from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constitutive import smooth_threshold_gate, three_phase_mobility
from .grid import REL_MF, REL_MM
from .physics import State, ThreePhaseMultiContinuumModel
from .wells import peaceman_well_index


REL_FW = 5
N_RELATIONS = 6


@dataclass
class GraphArrays:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_relation: np.ndarray
    reservoir_node_count: int
    well_node_count: int


def build_graph_arrays(
    model: ThreePhaseMultiContinuumModel,
    current: State,
    previous: State | None,
    dt_s: float,
    extrapolated_vector: np.ndarray | None = None,
    extrapolated_residual: np.ndarray | None = None,
) -> GraphArrays:
    grid = model.grid
    cfg = model.cfg
    n = grid.n_nodes
    nw = model.n_wells
    medium_one_hot = np.eye(4, dtype=np.float32)[grid.node_medium]
    if previous is None:
        dp_dt = np.zeros(n)
        dsw_dt = np.zeros(n)
        dsg_dt = np.zeros(n)
    else:
        dp_dt = (current.pressure_pa - previous.pressure_pa) / max(dt_s, 1e-30)
        dsw_dt = (current.water_saturation - previous.water_saturation) / max(dt_s, 1e-30)
        dsg_dt = (current.gas_saturation - previous.gas_saturation) / max(dt_s, 1e-30)

    permeability = model.permeability(current.pressure_pa)
    pore_volume = model._porosity * grid.node_volume
    if extrapolated_vector is None:
        extrapolated_vector = model.pack(current)
    extrapolated = model.unpack(extrapolated_vector)
    # v0.12 exposes the exact discrete defect at the candidate initial state.
    # A warm-start network must know which conservation equations are violated;
    # state variables alone do not identify a Newton descent direction.
    residual = (
        model.residual(extrapolated_vector, current, dt_s)
        if extrapolated_residual is None
        else np.asarray(extrapolated_residual, dtype=float)
    )
    if residual.shape != extrapolated_vector.shape:
        raise ValueError("precomputed extrapolated residual has an invalid shape")
    residual_p = residual[:n]
    residual_sw = residual[n : 2 * n]
    residual_sg = residual[2 * n : 3 * n]
    residual_well = residual[3 * n :]
    extrapolated_flux = model.edge_fluxes(extrapolated)
    row_conductance = np.zeros(n)
    np.add.at(
        row_conductance, grid.edge_i,
        np.abs(extrapolated_flux.transmissibility_m3_pa_s),
    )
    np.add.at(
        row_conductance, grid.edge_j,
        np.abs(extrapolated_flux.transmissibility_m3_pa_s),
    )
    diagonal_p = np.maximum(
        cfg.rock.compressibility_pa_inv * model.pressure_scale_pa
        + dt_s * row_conductance * model.pressure_scale_pa
        / np.maximum(pore_volume, 1e-20),
        1e-12,
    )
    approximate_p = -residual_p / diagonal_p
    approximate_sw = -residual_sw
    approximate_sg = -residual_sg
    reservoir_features = np.column_stack([
        current.pressure_pa / model.pressure_scale_pa,
        current.water_saturation,
        current.gas_saturation,
        dp_dt / model.pressure_scale_pa,
        dsw_dt,
        dsg_dt,
        extrapolated.pressure_pa / model.pressure_scale_pa,
        extrapolated.water_saturation,
        extrapolated.gas_saturation,
        np.log10(np.maximum(permeability, 1e-30)),
        model._porosity,
        medium_one_hot,
        np.full(n, np.log10(max(dt_s, 1.0))),
        np.full(n, cfg.microfracture.aperture_m / 1e-6),
        np.full(n, cfg.microfracture.surface_density_um_cm2 / 100.0),
        np.full(n, cfg.microfracture.penetration_ratio),
        np.log10(np.maximum(pore_volume, 1e-20)),
        np.zeros(n),
        np.zeros(n),
        residual_p,
        residual_sw,
        residual_sg,
        np.zeros(n),
        approximate_p,
        approximate_sw,
        approximate_sg,
        np.zeros(n),
    ])

    well_features = np.empty((0, reservoir_features.shape[1]), dtype=float)
    if nw:
        well_rows = []
        previous_well = None if previous is None else previous.well_pressure_pa
        for well_index, (well, node) in enumerate(zip(cfg.wells, model._well_node)):
            pwf = current.well_pressure_pa[well_index]
            dpwf_dt = 0.0 if previous_well is None else (
                pwf - previous_well[well_index]
            ) / max(dt_s, 1e-30)
            control_value = (
                well.value / model.pressure_scale_pa
                if well.control == "bhp" else well.value / 1.0e-5
            )
            well_rows.append([
                pwf / model.pressure_scale_pa,
                current.water_saturation[node],
                current.gas_saturation[node],
                dpwf_dt / model.pressure_scale_pa,
                dsw_dt[node],
                dsg_dt[node],
                extrapolated.well_pressure_pa[well_index] / model.pressure_scale_pa,
                extrapolated.water_saturation[node],
                extrapolated.gas_saturation[node],
                np.log10(max(permeability[node], 1e-30)),
                0.0,
                0.0, 0.0, 0.0, 1.0,
                np.log10(max(dt_s, 1.0)),
                cfg.microfracture.aperture_m / 1e-6,
                cfg.microfracture.surface_density_um_cm2 / 100.0,
                cfg.microfracture.penetration_ratio,
                -20.0,
                1.0 if well.control == "rate" else 0.0,
                control_value,
                0.0, 0.0, 0.0, residual_well[well_index],
                0.0, 0.0, 0.0, -residual_well[well_index],
            ])
        well_features = np.asarray(well_rows, dtype=float)
    node_features = np.vstack([reservoir_features, well_features]).astype(np.float32)

    flux = model.edge_fluxes(current)
    i = grid.edge_i.copy()
    j = grid.edge_j.copy()
    relation = grid.edge_relation.astype(np.int64).copy()
    delta_p = current.pressure_pa[i] - current.pressure_pa[j]
    gradient = np.abs(delta_p) / np.maximum(grid.edge_length, 1e-20)
    smooth_gate = np.ones(grid.n_edges)
    threshold_edge = np.isin(grid.edge_relation, (REL_MM, REL_MF))
    smooth_gate[threshold_edge] = smooth_threshold_gate(
        delta_p[threshold_edge],
        grid.edge_length[threshold_edge],
        cfg.microfracture.threshold_gradient_pa_m,
        kappa=12.0,
    )
    transmissibility = np.abs(flux.transmissibility_m3_pa_s)
    edge_length = grid.edge_length.copy()
    edge_area = grid.edge_area.copy()
    hard_gate = flux.hard_gate.copy()

    if nw:
        mob_o, mob_w, mob_g = three_phase_mobility(
            current.water_saturation, current.gas_saturation, cfg.fluid
        )
        permeability_now = model.permeability(current.pressure_pa)
        well_i, well_j, well_trans = [], [], []
        well_length, well_area, well_gradient = [], [], []
        for well_index, (well, node) in enumerate(zip(cfg.wells, model._well_node)):
            aperture = cfg.fracture.discrete_aperture_m if grid.node_medium[node] == 2 else None
            wi, _ = peaceman_well_index(
                permeability_now[node], cfg.grid.dz_m, cfg.grid.dx_m, cfg.grid.dy_m,
                well.radius_m, well.skin, aperture,
            )
            total_mob = mob_o[node] + mob_w[node] + mob_g[node]
            well_i.append(node)
            well_j.append(n + well_index)
            well_trans.append(abs(wi * total_mob))
            well_length.append(max(well.radius_m, 1e-6))
            well_area.append(2.0 * np.pi * well.radius_m * cfg.grid.dz_m)
            well_gradient.append(
                abs(current.pressure_pa[node] - current.well_pressure_pa[well_index])
                / max(well.radius_m, 1e-6)
            )
        i = np.concatenate([i, np.asarray(well_i, dtype=int)])
        j = np.concatenate([j, np.asarray(well_j, dtype=int)])
        relation = np.concatenate([relation, np.full(nw, REL_FW, dtype=np.int64)])
        transmissibility = np.concatenate([transmissibility, np.asarray(well_trans)])
        edge_length = np.concatenate([edge_length, np.asarray(well_length)])
        edge_area = np.concatenate([edge_area, np.asarray(well_area)])
        gradient = np.concatenate([gradient, np.asarray(well_gradient)])
        smooth_gate = np.concatenate([smooth_gate, np.ones(nw)])
        hard_gate = np.concatenate([hard_gate, np.ones(nw)])

    relation_one_hot = np.eye(N_RELATIONS, dtype=np.float32)[relation]
    edge_features = np.column_stack([
        np.log10(np.maximum(transmissibility, 1e-30)),
        np.log10(np.maximum(edge_length, 1e-12)),
        np.log10(np.maximum(edge_area, 1e-12)),
        gradient / max(cfg.microfracture.threshold_gradient_pa_m, 1.0),
        smooth_gate,
        hard_gate,
        relation_one_hot,
    ]).astype(np.float32)

    edge_index = np.column_stack([
        np.concatenate([i, j]),
        np.concatenate([j, i]),
    ]).T.astype(np.int64)
    return GraphArrays(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=np.concatenate([edge_features, edge_features], axis=0),
        edge_relation=np.concatenate([relation, relation]).astype(np.int64),
        reservoir_node_count=n,
        well_node_count=nw,
    )
