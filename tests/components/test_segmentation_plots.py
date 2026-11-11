import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.plots import plot_segmentation_ablation
from src.plots.segmentation import (
    plot_segmentation_curves,
)
from src.utils.environment import git_revision, software_versions


def test_segmentation_curves_and_environment(tmp_path):
    path = tmp_path / "curves.png"
    plot_segmentation_curves([1.0, 0.8], [0.9, 0.85], [0.2, 0.3], path)
    assert path.stat().st_size > 0
    with pytest.raises(ValueError, match="match in length"):
        plot_segmentation_curves([1.0], [0.9], [0.2, 0.3], path)
    assert software_versions()["torch"] == torch.__version__
    assert set(git_revision()) == {"commit", "dirty"}


def test_ablation_figures_match_saved_measurements(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[2] / "results/a2/ablation/comparisons.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    captured = {}

    def capture(figure, path):
        axis = figure.axes[0]
        if path.stem == "ablation_class_deltas":
            captured[path.stem] = np.array(axis.images[0].get_array())
        else:
            captured[path.stem] = [bar.get_width() for bar in axis.patches]

    monkeypatch.setattr(plot_segmentation_ablation, "save_figure", capture)
    monkeypatch.setattr("sys.argv", ["plot", "--input", str(source), "--output", str(tmp_path)])
    plot_segmentation_ablation.main()
    np.testing.assert_allclose(
        captured["ablation_scores"][:6], [100 * float(row["miou"]) for row in rows]
    )
    scores = np.array(
        [
            [float(row[f"iou_{label}"]) for label in plot_segmentation_ablation.CLASSES]
            for row in rows
        ]
    )
    np.testing.assert_allclose(captured["ablation_class_deltas"], 100 * (scores[1:] - scores[0]))
    np.testing.assert_allclose(
        captured["ablation_compute"], [float(row["training_seconds"]) / 60 for row in rows]
    )
    rows[1]["seed"] = "43"
    invalid = tmp_path / "mixed_seeds.csv"
    with invalid.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr("sys.argv", ["plot", "--input", str(invalid)])
    with pytest.raises(ValueError, match="exactly one seed"):
        plot_segmentation_ablation.main()
