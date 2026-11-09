"""Plot saved ablation comparisons without loading models or rerunning training.

Usage: python -m scripts.plots.plot_segmentation_ablation
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plots.common import save_figure
from src.plots.format import COLORS, DIVERGING, apply, text_color

VARIANTS = {
    "reference": "Reference",
    "frozen": "Frozen backbone",
    "no_augmentation": "No augmentation",
    "resnet101": "ResNet101",
    "ce_dice": "CE + Dice",
    "low_resolution": "Half resolution",
}
CLASSES = ("BW", "HD", "PF", "WR", "RO", "RI", "FV", "SR")


def main() -> None:
    """Render scores, class deltas and compute measurements for one paired seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/a2/ablation/comparisons.csv"))
    parser.add_argument("--output", type=Path, default=Path("web/images/a2/m3"))
    args = parser.parse_args()
    with args.input.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    seeds = {row["seed"] for row in rows}
    if len(seeds) != 1 or len(rows) != len(VARIANTS):
        raise ValueError("expected six paired configurations from exactly one seed")
    seed = seeds.pop()
    indexed = {row["run"]: row for row in rows}
    names = [f"suim_ablation_{variant}_seed{seed}" for variant in VARIANTS]
    if set(indexed) != set(names):
        raise ValueError("comparison rows do not match the six ablation configurations")
    ordered = [indexed[name] for name in names]
    fields = (
        "miou",
        "mean_dice",
        "delta_miou_pp",
        "delta_dice_pp",
        "training_seconds",
        "peak_training_memory_mb",
        "inference_ms_per_image",
        *(f"iou_{label}" for label in CLASSES),
    )
    data = {field: np.array([float(row[field]) for row in ordered]) for field in fields}
    if any(not np.isfinite(values).all() for values in data.values()):
        raise ValueError("comparison values must be finite")
    for metric, delta_field in (("miou", "delta_miou_pp"), ("mean_dice", "delta_dice_pp")):
        if not np.allclose(
            data[delta_field],
            100 * (data[metric] - data[metric][0]),
        ):
            raise ValueError(f"recorded {metric} deltas disagree with scores")
    labels = list(VARIANTS.values())
    positions = np.arange(len(labels))

    apply()
    figure, axis = plt.subplots(figsize=(7.1, 3.4), layout="constrained")
    for offset, metric, label, color in (
        (-0.18, "miou", "mIoU", COLORS[0]),
        (0.18, "mean_dice", "Mean Dice", COLORS[1]),
    ):
        bars = axis.barh(
            positions + offset, 100 * data[metric], height=0.32, label=label, color=color
        )
        axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set(yticks=positions, yticklabels=labels, xlim=(0, 85), xlabel="Validation score (%)")
    axis.invert_yaxis()
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, borderaxespad=0)
    save_figure(figure, args.output / "ablation_scores.png")
    plt.close(figure)

    apply("heatmap")
    class_scores = np.column_stack([data[f"iou_{label}"] for label in CLASSES])
    delta = 100 * (class_scores[1:] - class_scores[0])
    limit = max(float(np.abs(delta).max()), 1.0)
    figure, axis = plt.subplots(figsize=(7.1, 3.3), layout="constrained")
    image = axis.imshow(delta, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    axis.set(
        xticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticks=np.arange(len(labels) - 1),
        yticklabels=labels[1:],
        xlabel="SUIM class",
    )
    for row, column in np.ndindex(delta.shape):
        axis.text(
            column,
            row,
            f"{delta[row, column]:+.1f}",
            ha="center",
            va="center",
            color=text_color(DIVERGING((float(delta[row, column]) + limit) / (2 * limit))),
        )
    figure.colorbar(image, ax=axis, label="IoU change from reference (pp)")
    save_figure(figure, args.output / "ablation_class_deltas.png")
    plt.close(figure)

    apply()
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 3.4), sharey=True, layout="constrained")
    for axis, field, scale, title, unit in (
        (axes[0], "training_seconds", 1 / 60, "Training", "Minutes"),
        (axes[1], "peak_training_memory_mb", 1 / 1024, "Peak training memory", "GiB"),
        (axes[2], "inference_ms_per_image", 1, "Inference", "ms / image"),
    ):
        values = data[field] * scale
        if np.any(values <= 0):
            raise ValueError("compute measurements must be positive")
        bars = axis.barh(positions, values, color=COLORS[1])
        bars[0].set_color(COLORS[0])
        axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.set(title=title, xlabel=unit, xlim=(0, float(values.max()) * 1.35))
    axes[0].set(yticks=positions, yticklabels=labels)
    axes[0].invert_yaxis()
    save_figure(figure, args.output / "ablation_compute.png")
    plt.close(figure)


if __name__ == "__main__":
    main()
