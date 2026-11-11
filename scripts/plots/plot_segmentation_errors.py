"""Plot saved A2 validation errors without training or model inference.

Run from the repository root: python -m scripts.plots.plot_segmentation_errors
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.data.suim import CLASS_NAMES
from src.plots.common import plot_confusion_matrix, save_figure
from src.plots.format import COLORS, DIVERGING, apply

RUN = Path("results/a2/suim_ablation_reference_seed42")
OUTPUT = Path("web/images/a2/errors")


def main() -> None:
    """Compare regional scores and inspect individual saved validation predictions."""
    evidence = torch.load(RUN / "val_evidence.pt", map_location="cpu", weights_only=True)
    targets = evidence["targets"].numpy()
    predictions = evidence["predictions"].numpy()
    images = evidence["images"].numpy()
    ids = evidence["sample_ids"]
    summary = json.loads((RUN / "training_summary.json").read_text())
    assert evidence["split"] == "validation"
    assert targets.shape == predictions.shape == (len(ids), 240, 320)
    assert images.shape == (len(ids), 3, 240, 320)
    assert targets.max() < 8 and predictions.max() < 8
    counts = np.stack(
        [
            np.bincount(8 * target.astype(np.int64).ravel() + pred.ravel(), minlength=64).reshape(
                8, 8
            )
            for target, pred in zip(targets, predictions, strict=True)
        ]
    )
    pooled = counts.sum(0)
    diagonal = pooled.diagonal()
    pooled_iou = diagonal / (pooled.sum(0) + pooled.sum(1) - diagonal)
    np.testing.assert_allclose(
        pooled_iou, list(summary["validation"]["all"]["per_class_iou"].values())
    )
    intersection = counts.diagonal(axis1=1, axis2=2)
    union = counts.sum(1) + counts.sum(2) - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(union, dtype=float), where=union > 0)
    scores = iou.sum(1) / (union > 0).sum(1)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = [
        json.loads(Path("results/a2/suim_unet_seed42/training_summary.json").read_text()),
        json.loads((RUN / "training_summary.json").read_text()),
    ]
    apply()
    figure, axis = plt.subplots(layout="constrained")
    positions = np.arange(3)
    regions = ("all", "boundary", "interior")
    for index, (label, summary) in enumerate(zip(("U-Net", "DeepLabV3"), summaries, strict=True)):
        values = [100 * summary["validation"][region]["mean_iou"] for region in regions]
        bars = axis.bar(
            positions + (index - 0.5) * 0.34, values, width=0.34, color=COLORS[index], label=label
        )
        axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set(
        xticks=positions,
        xticklabels=("Whole mask", "Boundary", "Interior"),
        ylabel="Validation mIoU (%)",
        ylim=(0, 85),
    )
    axis.legend(loc="upper left", ncol=2)
    save_figure(figure, OUTPUT / "regional_iou.png")
    plt.close(figure)

    apply()
    figure, axis = plt.subplots(layout="constrained")
    axis.hist(100 * scores, bins=range(0, 101, 5), color=COLORS[0], edgecolor="white")
    median = float(np.median(scores))
    axis.axvline(
        100 * median, color=COLORS[3], linestyle="--", label=f"Median: {100 * median:.1f}%"
    )
    axis.set(xlim=(0, 100), xlabel="Per-image mIoU (%)", ylabel="Number of validation images")
    axis.legend()
    save_figure(figure, OUTPUT / "per_image_iou.png")
    plt.close(figure)

    palette = np.array([[(i >> bit) & 1 for bit in (2, 1, 0)] for i in range(8)], dtype=float)
    examples = {}
    robot_candidates = np.flatnonzero(counts.sum(2)[:, 4] > 0)
    for label, index in (
        ("best", int(np.argmax(scores))),
        ("worst", int(np.argmin(scores))),
        ("robot", int(robot_candidates[np.argmax(scores[robot_candidates])])),
    ):
        apply()
        figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.4), layout="constrained")
        for axis, panel, title in zip(
            axes,
            (
                images[index].transpose(1, 2, 0),
                palette[targets[index]],
                palette[predictions[index]],
            ),
            ("Input", "Ground truth", "Prediction"),
            strict=True,
        ):
            axis.imshow(panel)
            axis.set_title(title)
            axis.axis("off")
        save_figure(figure, OUTPUT / f"example_{label}.png")
        plt.close(figure)
        examples[label] = {"sample_id": ids[index], "miou": float(scores[index])}
    selected = {}
    apply()
    figure, axes = plt.subplots(2, 4, figsize=(7.1, 4.0), layout="constrained")
    for row, (label, class_index) in enumerate((("Plants", 2), ("Robots", 4))):
        areas = counts.sum(2)[:, class_index]
        candidates = np.flatnonzero(areas > 0)
        # Tied class IoUs select the largest target region, then the filename.
        index = min(candidates, key=lambda i: (iou[i, class_index], -areas[i], ids[i]))
        selected[label] = {
            "sample_id": ids[index],
            "class_iou": float(iou[index, class_index]),
            "target_pixel_share": float(areas[index] / targets[index].size),
        }
        panels = (
            images[index].transpose(1, 2, 0),
            palette[targets[index]],
            palette[predictions[index]],
            targets[index] != predictions[index],
        )
        for column, (axis, panel) in enumerate(zip(axes[row], panels, strict=True)):
            axis.imshow(panel, cmap=ListedColormap(["white", COLORS[3]]), vmin=0, vmax=1)
            title = ("Input", "Ground truth", "Prediction", "Errors")[column]
            axis.set_title(f"{label}: {ids[index]}\n{title}" if column == 0 else title)
            axis.axis("off")
    figure.legend(
        handles=[Patch(color=palette[i], label=label) for i, label in enumerate(CLASS_NAMES)],
        loc="outside lower center",
        ncol=8,
        columnspacing=1,
    )
    save_figure(figure, OUTPUT / "rare_class_failures.png")
    plt.close(figure)
    plot_confusion_matrix(
        counts.sum(0), CLASS_NAMES, OUTPUT / "confusion_matrix.png", cmap=DIVERGING
    )
    analysis = {
        "split": "validation",
        "run": RUN.name,
        "image_count": len(ids),
        "per_image_miou_convention": "Mean over classes present in target or prediction",
        "median_per_image_miou": median,
        "failure_selection": "Lowest target-class IoU; ties: largest target area, then filename",
        "failures": selected,
        "examples": examples,
        "per_image": [
            {"sample_id": name, "miou": float(score)}
            for name, score in zip(ids, scores, strict=True)
        ],
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in analysis.items() if key != "per_image"}, indent=2)
    )


if __name__ == "__main__":
    main()
