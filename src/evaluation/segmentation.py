"""Pixel metrics and bounded qualitative evidence for semantic segmentation.

Confusion matrices use target rows and prediction columns. Metrics aggregate pixels
across the complete split, include background, and assign zero to classes absent
from both targets and predictions. No smoothing changes the reported scores.
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
    """Count valid integer label pairs without silently casting invalid inputs.

    Args:
        targets: Integer labels of any shape, with values in [0, num_classes).
        predictions: Integer labels matching targets exactly in shape.
        num_classes: Number of classes, including background.

    Returns:
        CPU int64 matrix with target rows and prediction columns. Empty regions
        return an all-zero matrix so boundary/interior counts remain additive.

    Raises:
        ValueError: Shapes, class count, label types, or label ranges are invalid.
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
    """Compute IoU and Dice from additive counts, including absent classes as zero.

    Args:
        counts: Nonnegative square integer matrix, target rows/prediction columns.

    Returns:
        Per-class scores and unweighted means over every class, including background.

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
    """Evaluate full masks and ground-truth boundary/interior regions.

    Args:
        targets: Integer masks shaped (N, H, W); empty splits are rejected.
        predictions: Integer masks of the same shape. Convert model logits with
            logits.argmax(dim=1) before calling; resizing belongs to the data pipeline.
        num_classes: Number of dataset classes, including background.

    Returns:
        Metrics keyed by all, boundary, and interior. Region confusion matrices add
        to the full matrix. Empty regions have zero scores and zero pixel counts.
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
    """Select best, worst, and best RO-containing examples deterministically.

    Ranking uses image mIoU over classes present in either target or prediction,
    avoiding an absent-class penalty in qualitative selection. Ties select the
    first sample. RO selection is omitted if no target contains that class.
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
    """Evaluate model batches with pooled counts and bounded memory.

    Args:
        model: Returns logits (N,C,H,W), or a torchvision-style dictionary with
            logits under "out". Logit resolution must match the supplied masks.
        batches: Pairs of model-ready images and integer target masks (N,H,W).
        device: Model execution device; confusion counts accumulate on CPU.
        num_classes: Expected output channels, including background.

    Returns:
        Full, boundary, and interior metrics pooled across all batches. The model
        is moved to device; its previous training/evaluation mode is restored.

    Raises:
        ValueError: Batches are empty or model outputs have incompatible shapes.
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
