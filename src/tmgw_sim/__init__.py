"""TMGW numerical-simulation research code."""

from .config import CaseConfig, load_case
from .grid import MultiContinuumGrid, build_case_grid
from .physics import ThreePhaseMultiContinuumModel, TwoPhaseMultiContinuumModel
from .solver import NonlinearSolver, SolverResult

__all__ = [
    "CaseConfig",
    "load_case",
    "MultiContinuumGrid",
    "build_case_grid",
    "TwoPhaseMultiContinuumModel",
    "ThreePhaseMultiContinuumModel",
    "NonlinearSolver",
    "SolverResult",
]
