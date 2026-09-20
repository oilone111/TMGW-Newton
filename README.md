# TMGW-Newton

Research code accompanying the manuscript **“Accelerating Fully Implicit
Simulation of Multiscale Fractured Tight Reservoirs with a Threshold-Aware
Multi-Continuum Graph Warm Start.”**

TMGW-Newton couples a three-phase, fully implicit multi-continuum reservoir
model with a threshold-aware graph warm start. The graph model proposes only
the initial state for Newton iterations; the governing equations, nonlinear
residual, conservation checks, admissible-state projection, and convergence
criteria remain those of the numerical simulator.

## Main capabilities

- Matrix, continuum-fracture, discrete-main-fracture, and wellbore nodes.
- Pressure-sensitive matrix and fracture properties.
- Hard threshold-pressure-gradient activation in the physical residual.
- Matrix–fracture, fracture–fracture, and fracture–wellbore transfer.
- Forchheimer correction in discrete main fractures.
- Peaceman-equivalent fracture–wellbore coupling.
- Three-phase fully implicit finite-volume residuals.
- Damped Newton–Krylov solution with adaptive time stepping.
- Relation-specific graph message passing and a graph-level relaxation factor.
- Residual, mass-conservation, physical-bound, and fallback screening.

## Repository layout

```text
configs/                 Reproducible simulation and training configurations
data/                    Compressed synthetic multiscale datasets
examples/                Command-line simulation, training, and benchmark tools
src/tmgw_sim/            Numerical and graph-learning implementation
tests/                   Automated unit and integration tests
docs/                    Validation and field-input documentation
pyproject.toml            Package and dependency metadata
run_tests.py              Test-suite entry point
```

## Requirements

- Python 3.10 or newer.
- NumPy 1.26 or newer.
- SciPy 1.11 or newer.
- PyYAML 6.0 or newer.
- PyTorch 2.8.0 for TMGW training or inference.

The S0 and S1 numerical-solver modes do not require PyTorch.

## Installation

From a terminal opened in the repository root:

```bash
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the numerical core:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For graph-model training, plotting, and development tools:

```bash
python -m pip install -e ".[ai,viz,dev]"
```

PyTorch CPU wheels may alternatively be installed from the official PyTorch
index before installing the optional dependencies.

## Quick test

The supplied quick test runs all 15 automated tests and a complete 5×5
three-phase simulation in a temporary directory:

```bash
python examples/quick_test.py
```

A successful run ends with `QUICK_TEST_OK`. On the reference check, the unit
case completed 16 accepted time steps with no rollback; numerical timings vary
with hardware.

The two parts can also be run separately:

```bash
python run_tests.py
python examples/run_case.py configs/unit_5x5.yaml --mode S0 --output outputs/unit_s0
```

The simulation writes `solver_log.csv`, `newton_history.csv`, and `states.npz`.

## Solver modes

| Mode | Initial state | Learned graph model | Online screening |
|---|---|---:|---:|
| S0 | Previous converged time step | No | No |
| S1 | Linear temporal extrapolation | No | No |
| S2 | Graph-corrected extrapolated state | Yes | No |
| S3 | Threshold-aware graph warm start | Yes | Accept, blend, or fall back |

All modes use the same fully implicit residual and convergence criteria.

## Reproducing the synthetic workflow

### Extract the included datasets

To keep the web-uploadable repository compact, the 96 synthetic 21×21 cases
and 12 synthetic 41×41 scale-calibration cases are stored in
`data/synthetic_datasets.zip`. Extract them into the repository root before
training or running dataset diagnostics:

```bash
python -m zipfile -e data/synthetic_datasets.zip .
```

This creates `outputs/dataset_96_21x21_v2` and
`outputs/dataset_12_41x41_v18` with the paths expected by the example scripts.

### Run the 41×41 mechanism case

```bash
python examples/run_case.py configs/mechanism_41x41.yaml \
  --mode S0 --output outputs/mechanism_s0
```

### Regenerate the 21×21 training cases

The included dataset was generated with the following fixed seed:

```bash
python examples/generate_dataset.py configs/training_21x21.yaml \
  --cases 96 --seed 20260818 --workers 4 \
  --output outputs/dataset_96_21x21_v2
```

The compressed dataset archive also contains 12 synthetic 41×41
scale-calibration cases.

### Train the compact TMGW model

```bash
python examples/train_model.py configs/training_21x21.yaml \
  outputs/dataset_96_21x21_v2 \
  --extra-dataset outputs/dataset_12_41x41_v18 \
  --model-kind TMGW --backend torch --epochs 300 \
  --checkpoint outputs/tmgw_multiscale_v18_final.pt
```

Training and benchmark runtimes depend strongly on the processor, thread
settings, and optional backend. Record these details when comparing timings.
The included datasets permit retraining; a trained checkpoint is not included
in this release.

## Field-scale configuration

`configs/zhuang23_field_template.yaml` contains the 92×65×2 grid dimensions
and the averaged, non-confidential parameters reported in the manuscript. It is
a field-scale input template, not a complete history-matching model. Cellwise
properties, exact well completions, operational schedules, and observed
production series are required before field history matching can be reproduced.
Additional entries required by the configuration schema are explicitly treated
as illustrative placeholders rather than field measurements.
The missing inputs are listed in `docs/FIELD_DATA_REQUIREMENTS.md`.

The synthetic cases and automated tests are independent of the field template
and reproduce the numerical and warm-start workflow without field data.

## Validation scope

The automated tests cover threshold activation, effective matrix
permeability, Peaceman coupling, conservative internal fluxes, three-phase
state closure, wellbore unknowns, heterogeneous graph construction, relation
edges, dataset normalization, and the S3 safety interface. Verified reference
results are summarized in `docs/VALIDATION.md`.

## Citation

If this software contributes to a publication, cite the accompanying article
and this software release. Machine-readable author and version metadata are
provided in `CITATION.cff`. Add the final article DOI and repository DOI after
acceptance or archival release.

## License

The source code is released under the MIT License. See `LICENSE`.
