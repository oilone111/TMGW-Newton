from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import re

import yaml
import numpy as np


@dataclass
class GridConfig:
    nx: int
    ny: int
    nz: int
    dx_m: float
    dy_m: float
    dz_m: float


@dataclass
class RockConfig:
    porosity: float
    permeability_m2: float
    compressibility_pa_inv: float = 5.0e-10
    reference_pressure_pa: float = 49.5e6


@dataclass
class MicrofractureConfig:
    enabled: bool = True
    aperture_m: float = 15.1e-6
    aperture_reference_m: float = 15.1e-6
    surface_density_um_cm2: float = 234.3
    surface_density_reference_um_cm2: float = 234.3
    penetration_ratio: float = 0.4
    penetration_reference: float = 0.4
    stress_sensitivity_coefficient: float = 0.0094
    stress_sensitivity_exponent: float = -0.0891
    overburden_pressure_pa: float = 65.0e6
    boundary_layer_a: float = 0.15
    boundary_layer_b_pa_inv_m: float = 4.0e-5
    threshold_gradient_pa_m: float = 2.0e5


@dataclass
class FluidConfig:
    oil_viscosity_pa_s: float = 1.13e-3
    water_viscosity_pa_s: float = 0.55e-3
    gas_viscosity_pa_s: float = 0.018e-3
    oil_compressibility_pa_inv: float = 1.0e-9
    water_compressibility_pa_inv: float = 4.5e-10
    gas_compressibility_pa_inv: float = 1.0e-8
    oil_density_kg_m3: float = 767.0
    water_density_kg_m3: float = 1000.0
    gas_density_kg_m3: float = 120.0
    swc: float = 0.20
    sor: float = 0.20
    sgc: float = 0.0
    corey_water: float = 2.0
    corey_oil: float = 2.0
    corey_gas: float = 2.0


@dataclass
class FractureConfig:
    continuum_multiplier: float = 20.0
    continuum_porosity: float = 0.02
    continuum_transfer_shape: float = 2.0e-5
    discrete_cells: list[int] = field(default_factory=list)
    discrete_permeability_m2: float = 5.0e-11
    discrete_porosity: float = 0.10
    discrete_aperture_m: float = 3.0e-3
    forchheimer_beta_m_inv: float = 1.0e6
    mf_transfer_multiplier: float = 1.0
    fF_transfer_multiplier: float = 1.0


@dataclass
class WellConfig:
    name: str
    cell: int
    control: str
    value: float
    radius_m: float = 0.1
    skin: float = 0.0


@dataclass
class TimeConfig:
    total_time_s: float
    initial_dt_s: float
    min_dt_s: float
    max_dt_s: float
    adaptive: bool = True


@dataclass
class SolverConfig:
    mode: str = "S0"
    nonlinear_tolerance: float = 1.0e-7
    increment_tolerance: float = 1.0e-7
    mass_tolerance: float = 1.0e-6
    warm_start_mass_tolerance: float = 1.0e-3
    max_newton: int = 15
    gmres_tolerance: float = 1.0e-7
    gmres_maxiter: int = 120
    dense_jacobian_limit: int = 600
    armijo_c: float = 1.0e-4
    residual_accept: float = 0.8
    residual_reject: float = 1.2
    residual_use_max: float = 0.98
    # v0.18 calls the graph model only at the first step or close to a
    # Newton stopping boundary.  Zero disables this lightweight trigger.
    warm_start_trigger_residual: float = 6.0e-3
    warm_start_flat_probe_min_relaxation: float = 0.35
    warm_start_flat_probe_max_relaxation: float = 0.75
    warm_start_flat_ratio_tolerance: float = 5.0e-2


@dataclass
class InitialConfig:
    pressure_pa: float = 49.5e6
    water_saturation: float = 0.25
    gas_saturation: float = 0.0


@dataclass
class CaseConfig:
    name: str
    grid: GridConfig
    rock: RockConfig
    microfracture: MicrofractureConfig
    fluid: FluidConfig
    fracture: FractureConfig
    wells: list[WellConfig]
    time: TimeConfig
    solver: SolverConfig
    initial: InitialConfig


_NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _coerce_yaml_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _coerce_yaml_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_yaml_numbers(item) for item in value]
    if isinstance(value, str) and _NUMERIC.match(value.strip()):
        return float(value)
    return value


def _construct(data: dict[str, Any]) -> CaseConfig:
    data = _coerce_yaml_numbers(data)
    return CaseConfig(
        name=data["name"],
        grid=GridConfig(**data["grid"]),
        rock=RockConfig(**data["rock"]),
        microfracture=MicrofractureConfig(**data.get("microfracture", {})),
        fluid=FluidConfig(**data.get("fluid", {})),
        fracture=FractureConfig(**data.get("fracture", {})),
        wells=[WellConfig(**w) for w in data.get("wells", [])],
        time=TimeConfig(**data["time"]),
        solver=SolverConfig(**data.get("solver", {})),
        initial=InitialConfig(**data.get("initial", {})),
    )


def load_case(path: str | Path) -> CaseConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return _construct(data)


def save_case(config: CaseConfig, path: str | Path) -> None:
    def builtin(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: builtin(item) for key, item in value.items()}
        if isinstance(value, list):
            return [builtin(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(builtin(asdict(config)), handle, allow_unicode=True, sort_keys=False)
