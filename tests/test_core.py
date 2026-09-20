from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from tmgw_sim.config import load_case
from tmgw_sim.constitutive import (
    mixed_matrix_permeability,
    threshold_drive,
)
from tmgw_sim.grid import build_case_grid
from tmgw_sim.physics import ThreePhaseMultiContinuumModel
from tmgw_sim.solver import NonlinearSolver
from tmgw_sim.wells import peaceman_equivalent_radius, peaceman_well_index


ROOT = Path(__file__).resolve().parents[1]


class ConstitutiveTests(unittest.TestCase):
    def test_threshold_is_hard_in_physical_solver(self):
        dp = np.array([1.0e6, 4.0e6, 5.0e6, -5.0e6])
        length = np.full(4, 20.0)
        result = threshold_drive(dp, length, 2.0e5)
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0e6, -1.0e6])

    def test_microfracture_parameters_change_effective_permeability(self):
        cfg = load_case(ROOT / "configs" / "unit_5x5.yaml")
        permeability = mixed_matrix_permeability(
            1.1e-15, cfg.rock.porosity, 1.0, cfg.microfracture
        )
        self.assertGreater(float(permeability), 1.1e-15)
        cfg.microfracture.enabled = False
        permeability_off = mixed_matrix_permeability(
            1.1e-15, cfg.rock.porosity, 1.0, cfg.microfracture
        )
        self.assertEqual(float(permeability_off), 1.1e-15)


class PeacemanTests(unittest.TestCase):
    def test_square_grid_radius(self):
        re = peaceman_equivalent_radius(20.0, 20.0, 1.0e-15, 1.0e-15)
        self.assertAlmostEqual(re, 0.14 * math.sqrt(800.0), places=12)

    def test_discrete_fracture_well_index_matches_equation_569(self):
        permeability = 5.0e-11
        aperture = 3.0e-3
        wi, re = peaceman_well_index(
            permeability, 4.8, 20.0, 20.0, 0.1, aperture_m=aperture
        )
        expected = math.pi * aperture * permeability / math.log(re / 0.1)
        self.assertAlmostEqual(wi, expected, places=25)


class ConservationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_case(ROOT / "configs" / "unit_5x5.yaml")
        self.grid = build_case_grid(self.cfg)
        self.model = ThreePhaseMultiContinuumModel(self.cfg, self.grid)

    def test_internal_flux_is_antisymmetric(self):
        state = self.model.initial_state()
        state.pressure_pa += np.linspace(0.0, 8.0e6, self.grid.n_nodes)
        flux = self.model.edge_fluxes(state)
        divergence = np.zeros(self.grid.n_nodes)
        np.add.at(divergence, self.grid.edge_i, flux.total_m3_s)
        np.add.at(divergence, self.grid.edge_j, -flux.total_m3_s)
        self.assertAlmostEqual(float(np.sum(divergence)), 0.0, places=18)

    def test_static_closed_state_has_zero_residual(self):
        self.cfg.wells = []
        grid = build_case_grid(self.cfg)
        model = ThreePhaseMultiContinuumModel(self.cfg, grid)
        state = model.initial_state()
        residual = model.residual(model.pack(state), state, 3600.0)
        self.assertLess(float(np.linalg.norm(residual, ord=np.inf)), 1e-14)
        result = NonlinearSolver(model, self.cfg.solver).solve_timestep(state, 3600.0)
        self.assertTrue(result.converged)

    def test_three_phase_state_closure_and_vector_size(self):
        state = self.model.initial_state()
        state.gas_saturation[:] = 0.05
        projected = self.model.unpack(self.model.physical_projection(self.model.pack(state)))
        oil = 1.0 - projected.water_saturation - projected.gas_saturation
        self.assertTrue(np.all(oil >= self.cfg.fluid.sor - 1e-12))
        self.assertEqual(
            self.model.pack(projected).size,
            3 * self.grid.n_nodes + len(self.cfg.wells),
        )


if __name__ == "__main__":
    unittest.main()
