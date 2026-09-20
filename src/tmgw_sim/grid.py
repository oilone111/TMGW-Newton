from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CaseConfig


REL_MM = 0
REL_FF = 1
REL_FF_DISCRETE = 2
REL_MF = 3
REL_FD = 4


@dataclass
class MultiContinuumGrid:
    n_matrix: int
    n_continuum: int
    n_discrete: int
    cell_volume: np.ndarray
    node_volume: np.ndarray
    node_medium: np.ndarray
    node_base_cell: np.ndarray
    edge_i: np.ndarray
    edge_j: np.ndarray
    edge_relation: np.ndarray
    edge_length: np.ndarray
    edge_area: np.ndarray
    base_cell_centers: np.ndarray
    discrete_global_nodes: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.node_medium.size)

    @property
    def n_edges(self) -> int:
        return int(self.edge_i.size)


def _cartesian_neighbors(nx: int, ny: int, nz: int):
    def idx(i: int, j: int, k: int) -> int:
        return i + nx * (j + ny * k)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = idx(i, j, k)
                if i + 1 < nx:
                    yield c, idx(i + 1, j, k), 0
                if j + 1 < ny:
                    yield c, idx(i, j + 1, k), 1
                if k + 1 < nz:
                    yield c, idx(i, j, k + 1), 2


def build_case_grid(cfg: CaseConfig) -> MultiContinuumGrid:
    g = cfg.grid
    n = g.nx * g.ny * g.nz
    dx = np.array([g.dx_m, g.dy_m, g.dz_m], dtype=float)
    areas = np.array([g.dy_m * g.dz_m, g.dx_m * g.dz_m, g.dx_m * g.dy_m])
    base_volume = np.full(n, np.prod(dx), dtype=float)

    coords = np.empty((n, 3), dtype=float)
    for k in range(g.nz):
        for j in range(g.ny):
            for i in range(g.nx):
                c = i + g.nx * (j + g.ny * k)
                coords[c] = [(i + 0.5) * g.dx_m, (j + 0.5) * g.dy_m, (k + 0.5) * g.dz_m]

    discrete_cells = np.array(sorted(set(cfg.fracture.discrete_cells)), dtype=int)
    if np.any((discrete_cells < 0) | (discrete_cells >= n)):
        raise ValueError("discrete fracture cell index is outside the base grid")

    # Matrix and continuum-fracture nodes occupy every active base cell.
    node_medium = np.concatenate([
        np.zeros(n, dtype=np.int8),
        np.ones(n, dtype=np.int8),
        np.full(discrete_cells.size, 2, dtype=np.int8),
    ])
    node_base = np.concatenate([np.arange(n), np.arange(n), discrete_cells])
    node_volume = np.concatenate([
        base_volume,
        base_volume,
        base_volume[discrete_cells] * cfg.fracture.discrete_aperture_m / max(g.dz_m, 1e-12),
    ])

    ei: list[int] = []
    ej: list[int] = []
    rel: list[int] = []
    length: list[float] = []
    area: list[float] = []

    neighbors = list(_cartesian_neighbors(g.nx, g.ny, g.nz))
    for a, b, axis in neighbors:
        for offset, relation in ((0, REL_MM), (n, REL_FF)):
            ei.append(a + offset)
            ej.append(b + offset)
            rel.append(relation)
            length.append(dx[axis])
            area.append(areas[axis])

    # Matrix-continuum exchange edges are local, not geometrical grid faces.
    transfer_length = 0.5 * min(g.dx_m, g.dy_m, g.dz_m)
    transfer_area = base_volume[0] / max(transfer_length, 1e-12)
    for c in range(n):
        ei.append(c)
        ej.append(n + c)
        rel.append(REL_MF)
        length.append(transfer_length)
        area.append(transfer_area)

    # Discrete-fracture nodes connect to the continuum node in the same cell.
    discrete_global = np.arange(2 * n, 2 * n + discrete_cells.size, dtype=int)
    for local, c in enumerate(discrete_cells):
        ei.append(n + int(c))
        ej.append(2 * n + local)
        rel.append(REL_FD)
        length.append(0.5 * min(g.dx_m, g.dy_m))
        area.append(cfg.fracture.discrete_aperture_m * g.dz_m)

    # Consecutive selected cells form connected main-fracture edges.
    lookup = {int(c): int(2 * n + q) for q, c in enumerate(discrete_cells)}
    for a, b, axis in neighbors:
        if a in lookup and b in lookup:
            ei.append(lookup[a])
            ej.append(lookup[b])
            rel.append(REL_FF_DISCRETE)
            length.append(dx[axis])
            area.append(cfg.fracture.discrete_aperture_m * (g.dz_m if axis < 2 else min(g.dx_m, g.dy_m)))

    return MultiContinuumGrid(
        n_matrix=n,
        n_continuum=n,
        n_discrete=discrete_cells.size,
        cell_volume=base_volume,
        node_volume=node_volume,
        node_medium=node_medium,
        node_base_cell=node_base,
        edge_i=np.asarray(ei, dtype=int),
        edge_j=np.asarray(ej, dtype=int),
        edge_relation=np.asarray(rel, dtype=np.int8),
        edge_length=np.asarray(length, dtype=float),
        edge_area=np.asarray(area, dtype=float),
        base_cell_centers=coords,
        discrete_global_nodes=discrete_global,
    )

