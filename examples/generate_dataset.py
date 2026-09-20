from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from tmgw_sim.config import load_case, save_case
from tmgw_sim.diagnostics import save_run
from tmgw_sim.grid import build_case_grid
from tmgw_sim.physics import ThreePhaseMultiContinuumModel
from tmgw_sim.sampling import sample_cases
from tmgw_sim.solver import NonlinearSolver, run_schedule


MANIFEST_FIELDS = [
    "case_index", "case_name", "split", "facies", "porosity",
    "permeability_m2", "aperture_m", "surface_density_um_cm2",
    "penetration_ratio", "injection_rate_m3_s", "accepted_steps", "converged",
]


def _base_row(index, cfg, split: str) -> dict:
    return {
        "case_index": index,
        "case_name": cfg.name,
        "split": split,
        "facies": cfg.name.rsplit("_", 2)[-2],
        "porosity": cfg.rock.porosity,
        "permeability_m2": cfg.rock.permeability_m2,
        "aperture_m": cfg.microfracture.aperture_m,
        "surface_density_um_cm2": cfg.microfracture.surface_density_um_cm2,
        "penetration_ratio": cfg.microfracture.penetration_ratio,
        "injection_rate_m3_s": next(
            (well.value for well in cfg.wells if well.control == "rate"), 0.0
        ),
    }


def _completed_row(index, cfg, split: str, case_dir: Path) -> dict | None:
    required = (case_dir / "case.yaml", case_dir / "states.npz", case_dir / "solver_log.csv")
    if not all(path.exists() for path in required):
        return None
    with (case_dir / "solver_log.csv").open("r", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if not records or records[-1]["accepted"].lower() != "true":
        return None
    row = _base_row(index, cfg, split)
    row.update({
        "accepted_steps": sum(item["accepted"].lower() == "true" for item in records),
        "converged": True,
    })
    return row


def _simulate_case(index, cfg, split: str, output: str, resume: bool) -> dict:
    case_dir = Path(output) / f"case_{index:03d}"
    if resume:
        existing = _completed_row(index, cfg, split, case_dir)
        if existing is not None:
            existing["status"] = "reused"
            return existing
    grid = build_case_grid(cfg)
    model = ThreePhaseMultiContinuumModel(cfg, grid)
    solver = NonlinearSolver(model, cfg.solver)
    states, records = run_schedule(
        solver, model.initial_state(), cfg.time.total_time_s,
        cfg.time.initial_dt_s, cfg.time.min_dt_s, cfg.time.max_dt_s,
        cfg.time.adaptive,
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    save_case(cfg, case_dir / "case.yaml")
    save_run(case_dir, states, records, model)
    row = _base_row(index, cfg, split)
    row.update({
        "accepted_steps": len(states) - 1,
        "converged": bool(records and records[-1].accepted),
        "status": "computed",
    })
    return row


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows([
            {key: row[key] for key in MANIFEST_FIELDS}
            for row in sorted(rows, key=lambda item: int(item["case_index"]))
        ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    base = load_case(args.config)
    base.solver.mode = "S0"
    args.output.mkdir(parents=True, exist_ok=True)
    cases = sample_cases(base, args.cases, args.seed)
    if len(cases) < 3:
        raise ValueError("at least three complete cases are required")
    order = np.random.default_rng(args.seed).permutation(len(cases))
    n_train = int(round(len(cases) * 68 / 96))
    n_validation = int(round(len(cases) * 14 / 96))
    split_by_index = {}
    for rank, index in enumerate(order):
        split_by_index[int(index)] = (
            "train" if rank < n_train
            else "validation" if rank < n_train + n_validation
            else "test"
        )
    manifest: list[dict] = []
    jobs = [
        (index, cfg, split_by_index[index], str(args.output), not args.no_resume)
        for index, cfg in enumerate(cases)
    ]
    workers = max(1, args.workers)
    if workers == 1:
        iterator = (_simulate_case(*job) for job in jobs)
        for row in iterator:
            manifest.append(row)
            _write_manifest(args.output / "dataset_manifest.csv", manifest)
            print(row["case_index"], row["case_name"], row["accepted_steps"],
                  row["converged"], row["status"], flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_simulate_case, *job) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                manifest.append(row)
                _write_manifest(args.output / "dataset_manifest.csv", manifest)
                print(row["case_index"], row["case_name"], row["accepted_steps"],
                      row["converged"], row["status"], flush=True)


if __name__ == "__main__":
    main()
