"""Evaluate saved SUIM predictions without depending on unfinished model loaders.

Usage:
    python -m scripts.evaluation.eval_segmentation --evidence run.pt --output results/a2/run

Evidence is a weights-only torch dictionary containing integer targets/predictions
(N,H,W), split (validation or test), and optional display images (N,3,H,W), float
RGB in [0,1]. Images must be unnormalized. Test evidence is evaluated only with
--final-test, after configurations are fixed. Filenames/IDs may be included as
sample_ids, a list of strings. Store checkpoint/config provenance alongside evidence.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch

from src.data.suim import CLASS_NAMES
from src.evaluation.segmentation import evaluate_predictions, select_examples
from src.plots.common import (
    plot_confusion_matrix,
    save_figure,
)
from src.plots.format import apply


def main() -> None:
    """Validate evidence, write regional/per-class metrics, and render selected masks.

    Outputs:
        metrics.json: Split-wide counts, per-class scores, region pixel counts,
            selection indices, and explicit metric conventions.
        confusion_matrix.png: Pixel confusion counts with SUIM class labels.
        examples.png: Best/worst/RO target-prediction panels; RGB when supplied.

    Raises:
        ValueError: Missing/invalid evidence, display images, IDs, or test authorization.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    evidence = torch.load(args.evidence, map_location="cpu", weights_only=True)
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be a dictionary")
    split = evidence.get("split")
    if split not in ("validation", "test") or (split == "test" and not args.final_test):
        raise ValueError("use validation evidence, or --final-test for fixed test configurations")
    targets, predictions = evidence.get("targets"), evidence.get("predictions")
    if not isinstance(targets, torch.Tensor) or not isinstance(predictions, torch.Tensor):
        raise ValueError("targets and predictions must be tensors")
    metrics = evaluate_predictions(targets, predictions)
    selected = select_examples(targets, predictions)
    images = evidence.get("images")
    if images is not None and (
        not isinstance(images, torch.Tensor)
        or images.shape != (targets.shape[0], 3, *targets.shape[1:])
        or not images.is_floating_point()
        or not torch.isfinite(images).all()
        or images.min() < 0
        or images.max() > 1
    ):
        raise ValueError("images must be finite float RGB (N,3,H,W) in [0,1]")
    ids = evidence.get("sample_ids", [str(i) for i in range(len(targets))])
    if (
        not isinstance(ids, list)
        or len(ids) != len(targets)
        or any(not isinstance(item, str) for item in ids)
    ):
        raise ValueError("sample_ids must contain one string per image")
    record = {
        "split": split,
        "evidence": str(args.evidence),
        "conventions": {
            "classes": list(CLASS_NAMES),
            "absent_class_score": 0,
            "background_included": True,
            "ignored_labels": False,
            "boundary": "both sides of four-neighbour target transitions",
        },
        "metrics": {
            region: {
                "mean_iou": m.mean_iou,
                "mean_dice": m.mean_dice,
                "per_class_iou": dict(zip(CLASS_NAMES, m.per_class_iou.tolist(), strict=True)),
                "per_class_dice": dict(zip(CLASS_NAMES, m.per_class_dice.tolist(), strict=True)),
                "confusion_matrix": m.confusion_matrix.tolist(),
                "pixel_count": int(m.confusion_matrix.sum()),
            }
            for region, m in metrics.items()
        },
        "examples": {name: {"index": i, "sample_id": ids[i]} for name, i in selected.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    plot_confusion_matrix(
        metrics["all"].confusion_matrix.numpy(), CLASS_NAMES, args.output / "confusion_matrix.png"
    )
    palette = torch.tensor([[((i >> bit) & 1) for bit in (2, 1, 0)] for i in range(8)])
    columns = 3 if images is not None else 2
    apply()
    figure, axes = plt.subplots(
        len(selected), columns, squeeze=False, figsize=(4 * columns, 3 * len(selected))
    )
    for row, (name, index) in enumerate(selected.items()):
        panels = [palette[targets[index].long()], palette[predictions[index].long()]]
        titles = ["Target", "Prediction"]
        if images is not None:
            panels.insert(0, images[index].permute(1, 2, 0))
            titles.insert(0, "Input")
        for axis, panel, title in zip(axes[row], panels, titles, strict=True):
            axis.imshow(panel.numpy())
            axis.set_title(f"{name}: {ids[index]} — {title}")
            axis.axis("off")
    figure.legend(
        handles=[Patch(color=palette[i].numpy(), label=name) for i, name in enumerate(CLASS_NAMES)],
        loc="lower center",
        ncol=8,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(figure, args.output / "examples.png")
    plt.close(figure)


if __name__ == "__main__":
    main()
