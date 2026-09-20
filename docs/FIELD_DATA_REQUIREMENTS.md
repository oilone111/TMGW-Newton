# Zhuang 23 field-case input requirements

`configs/zhuang23_field_template.yaml` becomes a reproducible field
history-matching case only after the following machine-readable inputs are
provided.

| Category | Required input | Present public status |
|---|---|---|
| Grid geometry | DX, DY, DZ, TOPS, and ACTNUM for every cell | Only the 92×65×2 dimensions are available |
| Rock properties | PORO, PERMX, PERMY, PERMZ, and compressibility | Ranges and averaged values only |
| Pore/fracture parameters | Spatial pore-throat, aperture, density, and penetration-ratio fields | Averaged or sampling values only |
| Continuum fractures | Porosity, permeability, shape factors, and transfer coefficients | Spatial assignments unavailable |
| Main fractures | Coordinates, orientation, length, aperture, conductivity, and layer intersections | Average length and orientation range only |
| PVT | Pressure-dependent phase properties and solution gas ratio | Selected reported values only |
| Relative permeability | SWOF/SGOF or equivalent laboratory tables | Raw tables unavailable |
| Initial conditions | Pressure, saturation, and contact fields | Initial pressure only |
| Wells | Exact cell coordinates, completions, radius, and skin | Well count and average spacing only |
| Controls | Time-series injection, production, BHP, and control switches | Machine-readable schedule unavailable |
| Observations | Water cut, BHP, mean pressure, and breakthrough time | Curves only; raw values unavailable |

The template contains the averaged values stated in the manuscript, including
12.9% porosity, 1.1×10⁻³ μm² permeability, 4.8 m effective thickness, 0.86 μm
pore-throat radius, and 15.1 μm microfracture aperture. These values are
insufficient to reconstruct the full field model.

