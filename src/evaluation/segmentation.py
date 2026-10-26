"""Compute pixel metrics and select segmentation examples.

Notes:
    Confusion matrices use target rows and prediction columns.
    Metrics pool pixels across the split, include background, and use no smoothing.
    Classes absent from both targets and predictions receive zero scores.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SegmentationMetrics:
    """Split-level scores; class vectors follow the dataset's label order."""

    confusion_matrix: Tensor
    per_class_iou: Tensor
    per_class_dice: Tensor
    mean_iou: float
    mean_dice: float


def confusion_matrix(
    targets: Tensor,
    predictions: Tensor,
    *,
    num_classes: int = 8,
) -> Tensor:
    """Count valid integer label pairs.

    Args:
        targets: Integer labels in [0, num_classes), of any shape.
        predictions: Integer labels with the same shape as targets.
        num_classes: Class count including background.

    Returns:
        CPU int64 matrix with target rows and prediction columns.
        Empty inputs produce zero counts.

    Raises:
        ValueError: Invalid shapes, class count, label types, or label ranges.
    """
    if num_classes <= 0 or targets.shape != predictions.shape:
        raise ValueError("positive num_classes and matching label shapes are required")
    for labels in (targets, predictions):
        if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
            raise ValueError("labels must have integer dtype")
        if labels.numel() and (labels.min() < 0 or labels.max() >= num_classes):
            raise ValueError("labels must be in [0, num_classes); ignored labels are unsupported")
    target = targets.detach().cpu().to(torch.int64).flatten()
    prediction = predictions.detach().cpu().to(torch.int64).flatten()
    return torch.bincount(
        num_classes * target + prediction,
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)


def metrics_from_confusion(counts: Tensor) -> SegmentationMetrics:
    """Compute IoU and Dice from pooled counts.

    Args:
        counts: Nonnegative square integer matrix, with target rows.

    Returns:
        Per-class scores and unweighted means, including background.
        Absent classes receive zero scores.

    Raises:
        ValueError: Counts are malformed, negative, or not integers.
    """
    if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[0] != counts.shape[1]:
        raise ValueError("counts must be a nonempty square matrix")
    if (
        counts.dtype == torch.bool
        or counts.is_floating_point()
        or counts.is_complex()
        or (counts < 0).any()
    ):
        raise ValueError("counts must be nonnegative integers")
    values = counts.detach().cpu().to(torch.float64)
    true_positive = values.diag()
    total = values.sum(0) + values.sum(1)
    union = total - true_positive
    iou = true_positive / union.clamp_min(1)
    dice = 2 * true_positive / total.clamp_min(1)
    return SegmentationMetrics(
        counts.detach().cpu().clone(),
        iou,
        dice,
        iou.mean().item(),
        dice.mean().item(),
    )


def boundary_mask(targets: Tensor) -> Tensor:
    """Mark both sides of four-neighbour ground-truth class transitions.

    Args:
        targets: Label masks shaped (N, H, W). Image borders alone are not boundaries.

    Returns:
        Boolean mask of the same shape. No dilation or cross-image adjacency is used.
    """
    if targets.ndim != 3:
        raise ValueError("targets must have shape (N, H, W)")
    boundary = torch.zeros_like(targets, dtype=torch.bool)
    vertical = targets[:, 1:, :] != targets[:, :-1, :]
    horizontal = targets[:, :, 1:] != targets[:, :, :-1]
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    return boundary


def evaluate_predictions(
    targets: Tensor,
    predictions: Tensor,
    *,
    num_classes: int = 8,
) -> dict[str, SegmentationMetrics]:
    """Evaluate full masks, boundaries, and interiors.

    Args:
        targets: Nonempty integer masks shaped (N, H, W).
        predictions: Integer masks matching targets. Convert logits with argmax(dim=1).
        num_classes: Class count including background.

    Returns:
        Metrics keyed by all, boundary, and interior. Region counts sum to full counts.
        Empty regions have zero scores and pixel counts.

    Notes:
        Resize masks in the data pipeline before evaluation.
    """
    if targets.ndim != 3 or targets.numel() == 0:
        raise ValueError("targets must be a nonempty (N, H, W) tensor")
    full = confusion_matrix(targets, predictions, num_classes=num_classes)
    targets, predictions = targets.cpu(), predictions.cpu()
    boundary = boundary_mask(targets)
    return {
        "all": metrics_from_confusion(full),
        "boundary": metrics_from_confusion(
            confusion_matrix(
                targets[boundary],
                predictions[boundary],
                num_classes=num_classes,
            )
        ),
        "interior": metrics_from_confusion(
            confusion_matrix(
                targets[~boundary],
                predictions[~boundary],
                num_classes=num_classes,
            )
        ),
    }


def select_examples(
    targets: Tensor,
    predictions: Tensor,
    *,
    num_classes: int = 8,
    robot_class: int = 4,
) -> dict[str, int]:
    """Select best, worst, and best RO-containing examples.

    Notes:
        Ranks by image mIoU over classes present in either target or prediction.
        Ties select the first sample. Omits RO selection when no target contains RO.
    """
    evaluate_predictions(targets, predictions, num_classes=num_classes)
    if not 0 <= robot_class < num_classes:
        raise ValueError("robot_class must be a valid label")
    scores = []
    for target, prediction in zip(targets, predictions, strict=True):
        counts = confusion_matrix(target, prediction, num_classes=num_classes)
        present = counts.sum(0) + counts.sum(1) > 0
        scores.append(metrics_from_confusion(counts).per_class_iou[present].mean().item())
    selected = {
        "best": max(range(len(scores)), key=scores.__getitem__),
        "worst": min(range(len(scores)), key=scores.__getitem__),
    }
    robots = [i for i, target in enumerate(targets) if (target == robot_class).any()]
    if robots:
        selected["robot"] = max(robots, key=scores.__getitem__)
    return selected


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    *,
    num_classes: int = 8,
) -> dict[str, SegmentationMetrics]:
    """Evaluate batches with pooled counts and bounded memory.

    Args:
        model: Returns (N, C, H, W) logits or a dictionary with an "out" tensor.
            Spatial dimensions must match the masks.
        batches: Model-ready image and integer mask pairs; masks have shape (N, H, W).
        device: Execution device. Counts accumulate on CPU.
        num_classes: Output channel count including background.

    Returns:
        Pooled full-mask, boundary, and interior metrics.

    Notes:
        Moves the model to device and restores its previous training mode.

    Raises:
        ValueError: Empty batches or incompatible output shapes.
    """
    counts = {
        name: torch.zeros(num_classes, num_classes, dtype=torch.int64)
        for name in ("all", "boundary", "interior")
    }
    was_training = model.training
    model.to(device).eval()
    seen = 0
    try:
        for images, targets in batches:
            logits = model(images.to(device))
            if isinstance(logits, dict):
                logits = logits.get("out")
            if (
                not isinstance(logits, Tensor)
                or targets.ndim != 3
                or logits.shape != (targets.shape[0], num_classes, *targets.shape[1:])
                or not torch.isfinite(logits).all()
            ):
                raise ValueError("model must return finite logits (N,C,H,W) matching targets")
            regional = evaluate_predictions(targets, logits.argmax(1), num_classes=num_classes)
            for name, metrics in regional.items():
                counts[name] += metrics.confusion_matrix
            seen += len(targets)
    finally:
        model.train(was_training)
    if not seen:
        raise ValueError("batches produced no samples")
    return {name: metrics_from_confusion(matrix) for name, matrix in counts.items()}
