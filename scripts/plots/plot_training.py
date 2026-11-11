"""Replot saved segmentation history with the shared plotting template.

Run from the repository root: uv run python -m scripts.plots.plot_training
"""

import argparse
import json
import math
from pathlib import Path

from src.plots.segmentation import plot_segmentation_curves


def main() -> None:
    """Validate a saved history and render its loss and validation-mIoU curves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("results/a2/suim_ablation_reference_seed42")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/a2/Images/deeplabv3_ce_learning_curves.png")
    )
    args = parser.parse_args()
    history = json.loads((args.run_dir / "history.json").read_text())
    summary = json.loads((args.run_dir / "training_summary.json").read_text())
    assert history, "Training history is empty"
    assert [row["epoch"] for row in history] == list(range(1, len(history) + 1))
    series = [
        [float(row[key]) for row in history]
        for key in ("train_loss", "validation_loss", "validation_miou")
    ]
    assert all(math.isfinite(value) for values in series for value in values)
    assert all(value >= 0 for values in series[:2] for value in values)
    assert all(0 <= value <= 1 for value in series[2])
    best = max(history, key=lambda row: row["validation_miou"])
    assert best["epoch"] == summary["best_epoch"], "History and checkpoint summary disagree"
    plot_segmentation_curves(series[0], series[1], series[2], args.output)
    print(f"Saved {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
