# Validation record

## Automated checks

The release was verified from a clean source checkout with:

```bash
PYTHONPATH=src python run_tests.py
python examples/quick_test.py
```

All 15 automated tests passed. They cover threshold-pressure activation,
microfracture upscaling, Peaceman coupling, conservative flux assembly,
three-phase state closure, explicit wellbore variables, heterogeneous graph
construction, relation edges, dataset normalization, and the S3 safety path.

## 5×5 quick case

The S0 quick case completed 16 accepted time steps with no rollback. Reference
results from the release check were:

| Quantity | Value |
|---|---:|
| Total Newton iterations | 43 |
| Mean Newton iterations per accepted step | 2.6875 |
| Total linear iterations | 43 |
| Maximum final residual | 1.09×10⁻¹² |
| Maximum mass-conservation error | 2.02×10⁻¹⁰ |

These values verify the executable workflow; runtime is hardware-dependent and
the small case is not used to claim acceleration.

## Reproducibility boundary

The included synthetic datasets and configurations support verification of the
numerical model, training pipeline, and warm-start logic. The Zhuang 23 file is
only a field-input template because the manuscript does not contain cellwise
properties, exact well completions, schedules, or machine-readable observation
series. Consequently, the public package must not be described as reproducing
the field history match without those additional inputs.

