from __future__ import annotations

import math


def peaceman_equivalent_radius(dx: float, dy: float, kx: float, ky: float) -> float:
    """Peaceman equivalent radius for an anisotropic rectangular well block."""
    kx = max(kx, 1e-30)
    ky = max(ky, 1e-30)
    numerator = math.sqrt(math.sqrt(ky / kx) * dx**2 + math.sqrt(kx / ky) * dy**2)
    denominator = (ky / kx) ** 0.25 + (kx / ky) ** 0.25
    return 0.28 * numerator / max(denominator, 1e-30)


def peaceman_well_index(
    permeability_m2: float,
    thickness_m: float,
    dx_m: float,
    dy_m: float,
    radius_m: float,
    skin: float = 0.0,
    aperture_m: float | None = None,
) -> tuple[float, float]:
    """Return geometric well index [m^3] and equivalent radius [m].

    Phase mobility is multiplied outside this function.  When aperture_m is
    supplied, the discrete-fracture form pi*wF*KF/ln(re/rw) is used; otherwise
    the conventional 2*pi*K*h expression is used.
    """
    re = peaceman_equivalent_radius(dx_m, dy_m, permeability_m2, permeability_m2)
    if re <= radius_m:
        raise ValueError("equivalent well-block radius must exceed well radius")
    denom = math.log(re / radius_m) + skin
    if denom <= 0.0:
        raise ValueError("invalid Peaceman denominator")
    if aperture_m is not None:
        wi = math.pi * aperture_m * permeability_m2 / denom
    else:
        wi = 2.0 * math.pi * permeability_m2 * thickness_m / denom
    return wi, re

