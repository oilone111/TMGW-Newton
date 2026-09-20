from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tmgw_sim.config import load_case
from tmgw_sim.dataset import load_run_samples
from tmgw_sim.grid import build_case_grid
from tmgw_sim.physics import ThreePhaseMultiContinuumModel
from tmgw_sim.training import _normalization, _sample_difficulty, _target_matrix


def select_steps(samples, count: int):
    if count <= 0 or len(samples) <= count:
        return samples
    early_count = max(1, count // 2)
    early = list(range(min(early_count, len(samples))))
    tail_count = count - len(early)
    tail = [] if tail_count <= 0 else [
        int(round(value))
        for value in np.linspace(len(early), len(samples) - 1, tail_count)
    ]
    return [samples[index] for index in sorted(set(early + tail))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose v0.18 Newton-margin graph targets")
    parser.add_argument("config", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--steps-per-case", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("outputs/training_diagnostic_v18.json"))
    args = parser.parse_args()

    manifest_path = args.dataset / "dataset_manifest.csv"
    with manifest_path.open("r", encoding="utf-8") as handle:
        rows = {int(row["case_index"]): row for row in csv.DictReader(handle)}

    split_samples = {"train": [], "validation": [], "test": []}
    for case_dir in sorted(args.dataset.glob("case_*")):
        index = int(case_dir.name.rsplit("_", 1)[-1])
        row = rows[index]
        if row["converged"].lower() != "true":
            continue
        cfg = load_case(case_dir / "case.yaml") if (case_dir / "case.yaml").exists() else load_case(args.config)
        model = ThreePhaseMultiContinuumModel(cfg, build_case_grid(cfg))
        with np.load(case_dir / "states.npz") as arrays:
            available = int(arrays["pressure_pa"].shape[0] - 1)
        placeholder = list(range(available))
        selected = select_steps(placeholder, args.steps_per_case)
        split_samples[row["split"]].extend(
            load_run_samples(case_dir, model, step_indices=selected)
        )

    if not split_samples["train"]:
        raise RuntimeError("training split is empty")
    normalization = _normalization(split_samples["train"], "TMGW")
    scale = np.asarray(normalization["output_scale"], dtype=float)
    report = {
        "version": "0.18",
        "teacher_mode": "newton_margin",
        "steps_per_case": args.steps_per_case,
        "output_scale": scale.tolist(),
        "splits": {},
        "recommended": {
            "epochs": 150,
            "samples_per_epoch": 320,
            "learning_rate": 2.0e-4,
            "state_weight": 0.0,
            "residual_weight": 0.0,
            "direction_weight": 0.0,
            "mass_weight": 0.0,
            "active_node_boost": 4.0,
            "hard_sample_fraction": 0.55,
            "amplitude_weight": 0.0,
            "relaxation_weight": 1.0,
            "hard_step_boost": 1.0,
            "target_projection": 0.50,
            "curriculum_epochs": 1,
            "online_validation_samples": 32,
            "online_validation_interval": 10,
        },
    }
    for split, samples in split_samples.items():
        difficulties = np.asarray([_sample_difficulty(sample, scale) for sample in samples])
        target_matrices = [_target_matrix(sample) for sample in samples]
        nonzero_fraction = []
        physics_ratios = []
        for target in target_matrices:
            scaled = np.abs(target / scale)
            nonzero_fraction.append(float(np.mean(scaled > 1.0e-3)))
        for sample in samples:
            graph = sample.graph
            n = graph.reservoir_node_count
            nw = graph.well_node_count
            physics_correction = 0.5 * np.concatenate([
                graph.node_features[:n, 26],
                graph.node_features[:n, 27],
                graph.node_features[:n, 28],
                graph.node_features[n : n + nw, 29],
            ])
            model = sample.model
            current = sample.current_state
            if model is None or current is None:
                continue
            base = np.linalg.norm(
                model.residual(sample.extrapolated_state, current, sample.dt_s), ord=np.inf
            )
            candidate = model.warm_start_well_projection(
                sample.extrapolated_state + physics_correction
            )
            improved = np.linalg.norm(
                model.residual(candidate, current, sample.dt_s), ord=np.inf
            )
            physics_ratios.append(float(improved / max(base, 1.0e-30)))
        report["splits"][split] = {
            "samples": len(samples),
            "baseline_newton_quantiles": {
                name: float(value) for name, value in zip(
                    ("min", "median", "q75", "q90", "max"),
                    np.quantile(
                        [sample.baseline_newton_iterations for sample in samples],
                        [0.0, 0.5, 0.75, 0.9, 1.0],
                    ),
                )
            },
            "difficulty_quantiles": {
                name: float(value) for name, value in zip(
                    ("min", "median", "q75", "q90", "max"),
                    np.quantile(difficulties, [0.0, 0.5, 0.75, 0.9, 1.0]),
                )
            },
            "mean_active_target_fraction": float(np.mean(nonzero_fraction)),
            "half_physics_defect_residual_ratio": {
                "median": float(np.median(physics_ratios)),
                "q75": float(np.quantile(physics_ratios, 0.75)),
                "improved_fraction": float(np.mean(np.asarray(physics_ratios) < 1.0)),
            },
            "newton_margin_teacher": {
                "relaxation_quantiles": {
                    name: float(value) for name, value in zip(
                        ("min", "median", "q75", "q90", "max"),
                        np.quantile(
                            [sample.target_relaxation for sample in samples],
                            [0.0, 0.5, 0.75, 0.9, 1.0],
                        ),
                    )
                },
                "residual_ratio_median": float(np.median([
                    sample.target_relaxation_residual_ratio for sample in samples
                ])),
                "residual_ratio_q75": float(np.quantile([
                    sample.target_relaxation_residual_ratio for sample in samples
                ], 0.75)),
                "relaxation_counts": {
                    f"{value:.3f}": int(sum(
                        abs(sample.target_relaxation - value) < 1.0e-12
                        for sample in samples
                    ))
                    for value in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
                },
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
