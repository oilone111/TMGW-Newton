"""Run the editor-facing quick verification from a source checkout."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    env = os.environ.copy()
    source_dir = str(ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_dir, env.get("PYTHONPATH", "")) if part
    )

    run([sys.executable, "run_tests.py"], env)
    with tempfile.TemporaryDirectory(prefix="tmgw_quick_test_") as directory:
        output = Path(directory) / "unit_s0"
        run(
            [
                sys.executable,
                "examples/run_case.py",
                "configs/unit_5x5.yaml",
                "--mode",
                "S0",
                "--output",
                str(output),
            ],
            env,
        )
        required = ("solver_log.csv", "newton_history.csv", "states.npz")
        missing = [name for name in required if not (output / name).is_file()]
        if missing:
            raise RuntimeError(f"Quick test did not create: {', '.join(missing)}")

    print("QUICK_TEST_OK")


if __name__ == "__main__":
    main()

