from __future__ import annotations

import argparse
import json
from pathlib import Path

from tmgw_sim.config import load_case
from tmgw_sim.diagnostics import save_run, summarize
from tmgw_sim.grid import build_case_grid
from tmgw_sim.network import TorchCorrectionPredictor
from tmgw_sim.jax_network import JaxCorrectionPredictor
from tmgw_sim.physics import ThreePhaseMultiContinuumModel
from tmgw_sim.solver import NonlinearSolver, run_schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=["S0", "S1", "S2", "S3"], default=None)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    args = parser.parse_args()

    cfg = load_case(args.config)
    if args.mode:
        cfg.solver.mode = args.mode
    predictor = None
    if args.checkpoint:
        predictor = (
            JaxCorrectionPredictor(args.checkpoint)
            if args.checkpoint.suffix == ".npz"
            else TorchCorrectionPredictor(args.checkpoint)
        )
    grid = build_case_grid(cfg)
    model = ThreePhaseMultiContinuumModel(cfg, grid)
    solver = NonlinearSolver(model, cfg.solver, predictor)
    initial_state = model.initial_state()
    if predictor is not None:
        predictor.predict_correction(
            model, initial_state, None, cfg.time.initial_dt_s, model.pack(initial_state)
        )
    def report(item):
        result = item.result
        print(
            f"t={item.time_s / 86400.0:.2f} d  dt={item.dt_s / 86400.0:.2f} d  "
            f"accepted={item.accepted}  N={result.iterations}  K={result.linear_iterations}  "
            f"R={result.final_residual:.3e}  mass={result.final_mass_error:.3e}",
            flush=True,
        )
    states, records = run_schedule(
        solver,
        initial_state,
        cfg.time.total_time_s,
        cfg.time.initial_dt_s,
        cfg.time.min_dt_s,
        cfg.time.max_dt_s,
        cfg.time.adaptive,
        progress=report,
    )
    save_run(args.output, states, records, model)
    print(json.dumps(summarize(records), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
