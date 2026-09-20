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
from tmgw_sim.jax_training import train_jax_network


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--extra-dataset", type=Path, action="append", default=[])
    parser.add_argument("--extra-sample-repeat", type=int, default=3)
    parser.add_argument(
        "--teacher-mode",
        choices=["newton_margin", "residual_minimum"],
        default="newton_margin",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/tmgw.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--model-kind", choices=["GNN", "HGNN", "TMGW"], default="TMGW")
    parser.add_argument("--backend", choices=["jax", "torch"], default="jax")
    parser.add_argument("--steps-per-case", type=int, default=12)
    parser.add_argument(
        "--temporal-sampling", choices=["transient", "uniform"], default="transient"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--state-weight", type=float, default=0.0)
    parser.add_argument("--residual-weight", type=float, default=0.0)
    parser.add_argument("--direction-weight", type=float, default=0.0)
    parser.add_argument("--mass-weight", type=float, default=0.0)
    parser.add_argument("--active-node-boost", type=float, default=4.0)
    parser.add_argument("--hard-sample-fraction", type=float, default=0.55)
    parser.add_argument("--amplitude-weight", type=float, default=0.0)
    parser.add_argument("--hard-step-boost", type=float, default=1.0)
    parser.add_argument("--target-projection", type=float, default=0.50)
    parser.add_argument("--curriculum-epochs", type=int, default=1)
    parser.add_argument("--online-validation-samples", type=int, default=32)
    parser.add_argument("--online-validation-interval", type=int, default=10)
    parser.add_argument("--relaxation-weight", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_case(args.config)
    def split_directories(dataset_path: Path) -> tuple[list[Path], list[Path]]:
        case_dirs = sorted(
            p for p in dataset_path.iterdir() if (p / "states.npz").exists()
        )
        if len(case_dirs) < 3:
            raise RuntimeError(
                f"{dataset_path}: at least three complete simulation cases are required"
            )
        manifest_path = dataset_path / "dataset_manifest.csv"
        if not manifest_path.exists():
            raise RuntimeError(
                f"{dataset_path}: dataset_manifest.csv is required to prevent leakage"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest_rows = {
                int(row["case_index"]): row for row in csv.DictReader(handle)
            }

        def is_usable(path: Path, split: str) -> bool:
            row = manifest_rows[int(path.name.rsplit("_", 1)[-1])]
            return row["split"] == split and row["converged"].lower() == "true"

        train = [path for path in case_dirs if is_usable(path, "train")]
        validation = [
            path for path in case_dirs if is_usable(path, "validation")
        ]
        if not train or not validation:
            raise RuntimeError(
                f"{dataset_path}: training and validation groups must both be non-empty"
            )
        return train, validation

    train_dirs, validation_dirs = split_directories(args.dataset)
    def samples_for(path):
        case_cfg = load_case(path / "case.yaml") if (path / "case.yaml").exists() else cfg
        case_model = ThreePhaseMultiContinuumModel(case_cfg, build_case_grid(case_cfg))
        with np.load(path / "states.npz") as arrays:
            available = int(arrays["pressure_pa"].shape[0] - 1)
        if args.steps_per_case <= 0 or available <= args.steps_per_case:
            return load_run_samples(
                path, case_model, teacher_mode=args.teacher_mode
            )
        if args.temporal_sampling == "uniform":
            indices = sorted(set(
                int(round(value))
                for value in np.linspace(0, available - 1, args.steps_per_case)
            ))
        else:
            early_count = max(1, args.steps_per_case // 2)
            early = list(range(min(early_count, available)))
            remaining_count = args.steps_per_case - len(early)
            tail = [] if remaining_count <= 0 else [
                int(round(value))
                for value in np.linspace(len(early), available - 1, remaining_count)
            ]
            indices = sorted(set(early + tail))
        return load_run_samples(
            path,
            case_model,
            step_indices=indices,
            teacher_mode=args.teacher_mode,
        )

    train_samples = [s for path in train_dirs for s in samples_for(path)]
    validation_samples = [s for path in validation_dirs for s in samples_for(path)]
    for extra_dataset in args.extra_dataset:
        extra_train_dirs, extra_validation_dirs = split_directories(extra_dataset)
        extra_train = [
            sample for path in extra_train_dirs for sample in samples_for(path)
        ]
        extra_validation = [
            sample for path in extra_validation_dirs for sample in samples_for(path)
        ]
        train_samples.extend(
            extra_train * max(1, int(args.extra_sample_repeat))
        )
        validation_samples.extend(extra_validation)
    print(
        "training_samples", len(train_samples),
        "validation_samples", len(validation_samples),
        "teacher", args.teacher_mode,
        "extra_datasets", len(args.extra_dataset),
        flush=True,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "jax":
        trainer = train_jax_network
    else:
        from tmgw_sim.training import train_network
        trainer = train_network
    trainer_kwargs = {
        "epochs": args.epochs,
        "model_kind": args.model_kind,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
    }
    if args.backend == "jax":
        trainer_kwargs["samples_per_epoch"] = args.samples_per_epoch
        trainer_kwargs["validation_interval"] = args.validation_interval
        trainer_kwargs["initial_checkpoint"] = args.resume
    if args.backend == "torch":
        trainer_kwargs["device"] = args.device
        trainer_kwargs["samples_per_epoch"] = args.samples_per_epoch
        trainer_kwargs["validation_interval"] = args.validation_interval
        trainer_kwargs["initial_checkpoint"] = args.resume
        trainer_kwargs.update({
            "state_weight": args.state_weight,
            "residual_weight": args.residual_weight,
            "direction_weight": args.direction_weight,
            "mass_weight": args.mass_weight,
            "active_node_boost": args.active_node_boost,
            "hard_sample_fraction": args.hard_sample_fraction,
            "amplitude_weight": args.amplitude_weight,
            "hard_step_boost": args.hard_step_boost,
            "target_projection": args.target_projection,
            "curriculum_epochs": args.curriculum_epochs,
            "online_validation_samples": args.online_validation_samples,
            "online_validation_interval": args.online_validation_interval,
            "relaxation_weight": args.relaxation_weight,
        })
    history = trainer(
        train_samples, validation_samples, args.checkpoint, **trainer_kwargs,
    )
    history_path = args.checkpoint.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    args.checkpoint.with_suffix(".partial.history.json").unlink(missing_ok=True)
    print("epochs", len(history["train"]), "best_validation", min(history["validation"]))


if __name__ == "__main__":
    main()
