"""Run the same checks locally and in CI."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    environment = {**os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1"}
    commands = (
        ("ruff", "check", "src", "scripts", "tests"),
        ("ruff", "format", "--check", "src", "scripts", "tests"),
        ("ty", "check"),
        ("pytest", "-q"),
    )
    for command in commands:
        print("Running " + " ".join(command), flush=True)
        subprocess.run([sys.executable, "-m", *command], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
