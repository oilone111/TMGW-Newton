from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.stats import qmc

from .config import CaseConfig


FACIES_RANGES = {
    "I": {
        "porosity": (0.120, 0.163),
        "permeability_m2": (0.9e-15, 1.5e-15),
        "aperture_m": (20e-6, 40e-6),
        "surface_density_um_cm2": (230.0, 350.0),
        "penetration_ratio": (0.4, 0.8),
    },
    "II": {
        "porosity": (0.105, 0.145),
        "permeability_m2": (0.4e-15, 1.0e-15),
        "aperture_m": (15e-6, 30e-6),
        "surface_density_um_cm2": (170.0, 290.0),
        "penetration_ratio": (0.2, 0.6),
    },
    "III": {
        "porosity": (0.086, 0.120),
        "permeability_m2": (0.1e-15, 0.4e-15),
        "aperture_m": (5e-6, 15e-6),
        "surface_density_um_cm2": (50.0, 230.0),
        "penetration_ratio": (0.0, 0.2),
    },
}

FACIES_RATE_MULTIPLIER = {"I": 1.0, "II": 0.75, "III": 0.35}


def sample_cases(
    base: CaseConfig,
    n_cases: int,
    seed: int = 20260818,
) -> list[CaseConfig]:
    """Class-conditioned Latin-hypercube sampling from measured core ranges."""
    if n_cases < 1:
        return []
    sampler = qmc.LatinHypercube(d=5, seed=seed)
    unit = sampler.random(n_cases)
    facies_names = ("I", "II", "III")
    cases: list[CaseConfig] = []
    for index, row in enumerate(unit):
        facies = facies_names[index % 3]
        ranges = FACIES_RANGES[facies]
        values = {}
        for value, (key, bounds) in zip(row, ranges.items()):
            values[key] = bounds[0] + value * (bounds[1] - bounds[0])
        case = deepcopy(base)
        case.name = f"{base.name}_{facies}_{index:03d}"
        case.rock.porosity = values["porosity"]
        case.rock.permeability_m2 = values["permeability_m2"]
        case.microfracture.aperture_m = values["aperture_m"]
        case.microfracture.surface_density_um_cm2 = values["surface_density_um_cm2"]
        case.microfracture.penetration_ratio = values["penetration_ratio"]
        for well in case.wells:
            if well.control == "rate":
                well.value *= FACIES_RATE_MULTIPLIER[facies]
        cases.append(case)
    return cases
