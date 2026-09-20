from __future__ import annotations

import numpy as np

from .config import FluidConfig, MicrofractureConfig


def effective_saturation(sw: np.ndarray, fluid: FluidConfig) -> np.ndarray:
    """Backward-compatible water effective saturation."""
    width = max(1.0 - fluid.swc - fluid.sor - fluid.sgc, 1e-12)
    return np.clip((sw - fluid.swc) / width, 0.0, 1.0)


def relative_permeability(sw: np.ndarray, fluid: FluidConfig) -> tuple[np.ndarray, np.ndarray]:
    se = effective_saturation(sw, fluid)
    krw = se ** fluid.corey_water
    kro = (1.0 - se) ** fluid.corey_oil
    return kro, krw


def phase_mobility(sw: np.ndarray, fluid: FluidConfig) -> tuple[np.ndarray, np.ndarray]:
    kro, krw = relative_permeability(sw, fluid)
    return kro / fluid.oil_viscosity_pa_s, krw / fluid.water_viscosity_pa_s


def three_phase_relative_permeability(
    sw: np.ndarray,
    sg: np.ndarray,
    fluid: FluidConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized Corey closure for oil, water and gas.

    Oil saturation is closed by ``so=1-sw-sg``.  The function is deliberately
    table-replaceable: field SCAL tables can later replace this analytical
    closure without changing the residual or graph interfaces.
    """
    width = max(1.0 - fluid.swc - fluid.sor - fluid.sgc, 1e-12)
    sew = np.clip((sw - fluid.swc) / width, 0.0, 1.0)
    seg = np.clip((sg - fluid.sgc) / width, 0.0, 1.0)
    seo = np.clip(1.0 - sew - seg, 0.0, 1.0)
    kro = seo ** fluid.corey_oil
    krw = sew ** fluid.corey_water
    krg = seg ** fluid.corey_gas
    return kro, krw, krg


def three_phase_mobility(
    sw: np.ndarray,
    sg: np.ndarray,
    fluid: FluidConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kro, krw, krg = three_phase_relative_permeability(sw, sg, fluid)
    return (
        kro / fluid.oil_viscosity_pa_s,
        krw / fluid.water_viscosity_pa_s,
        krg / fluid.gas_viscosity_pa_s,
    )


def fluid_density(
    pressure: np.ndarray,
    reference_pressure: float,
    density_ref: float,
    compressibility: float,
) -> np.ndarray:
    return density_ref * np.exp(compressibility * (pressure - reference_pressure))


def microfracture_porosity(cfg: MicrofractureConfig) -> float:
    """Microfracture porosity phi_f=m*h*w from surface density and aperture."""
    if not cfg.enabled:
        return 0.0
    trace_length_per_area_m_inv = cfg.surface_density_um_cm2 * 1.0e-2
    return max(trace_length_per_area_m_inv * cfg.aperture_m, 0.0)


def microfracture_permeability(cfg: MicrofractureConfig) -> float:
    """Parallel-plate cubic-law permeability K_f=phi_f*w^2/12."""
    phi_f = microfracture_porosity(cfg)
    return phi_f * cfg.aperture_m**2 / 12.0


def mixed_matrix_permeability(
    pore_permeability_m2: np.ndarray | float,
    total_porosity: float,
    boundary_factor: np.ndarray | float,
    cfg: MicrofractureConfig,
) -> np.ndarray:
    """Pore-microfracture series/parallel upscaling relation preceding Eq. (4.45)."""
    kp = np.asarray(pore_permeability_m2, dtype=float)
    xi = np.asarray(boundary_factor, dtype=float)
    if not cfg.enabled:
        return kp.copy()
    phi_f = microfracture_porosity(cfg)
    kf = microfracture_permeability(cfg)
    gamma = np.clip(cfg.penetration_ratio, 0.0, 1.0)
    numerator = phi_f * kf * kp
    denominator = gamma * phi_f * kp * xi + total_porosity * (1.0 - gamma) * kf
    return kp + numerator / np.maximum(denominator, 1e-40)


def stress_sensitivity_factor(
    pressure: np.ndarray,
    permeability_m2: float,
    cfg: MicrofractureConfig,
) -> np.ndarray:
    if not cfg.enabled:
        return np.ones_like(pressure)
    # Equation (5.1) uses permeability in 10^-3 um^2 (approximately mD).
    k_md_scale = max(permeability_m2 / 1.0e-15, 1e-12)
    effective_stress_mpa = np.maximum(cfg.overburden_pressure_pa - pressure, 0.0) / 1.0e6
    exponent = -cfg.stress_sensitivity_coefficient * k_md_scale ** cfg.stress_sensitivity_exponent
    return np.exp(exponent * effective_stress_mpa)


def boundary_layer_factor(gradient_pa_m: np.ndarray, cfg: MicrofractureConfig) -> np.ndarray:
    if not cfg.enabled:
        return np.ones_like(gradient_pa_m)
    return (1.0 - cfg.boundary_layer_a * np.exp(-cfg.boundary_layer_b_pa_inv_m * np.abs(gradient_pa_m))) ** 4


def threshold_drive(delta_p: np.ndarray, length: np.ndarray, gradient_pa_m: float) -> np.ndarray:
    threshold = gradient_pa_m * length
    return np.sign(delta_p) * np.maximum(np.abs(delta_p) - threshold, 0.0)


def smooth_threshold_gate(
    delta_p: np.ndarray,
    length: np.ndarray,
    gradient_pa_m: float,
    kappa: float = 12.0,
    eps: float = 1e-12,
) -> np.ndarray:
    scale = gradient_pa_m * length
    ratio = np.abs(delta_p) / np.maximum(scale, eps)
    z = np.clip(kappa * (ratio - 1.0), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))
