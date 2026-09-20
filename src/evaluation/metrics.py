"""Classification metric interfaces."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClassificationMetrics:
    """Prediction-derived classification metrics."""

    accuracy: float
    macro_f1: float
    confusion_matrix: torch.Tensor


def compute_classification_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    num_classes: int,
) -> ClassificationMetrics:
    """Compute the assignment's classification metrics."""
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")

    targets = targets.long()
    predictions = predictions.long()

    confusion = _confusion_matrix(targets, predictions, num_classes)
    accuracy = (predictions == targets).float().mean().item()
    macro_f1 = _macro_f1(confusion)

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        confusion_matrix=confusion,
    )


def _confusion_matrix(
    targets: torch.Tensor, predictions: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Build a [num_classes, num_classes] matrix; rows = true label, cols = predicted."""
    flat_indices = targets * num_classes + predictions
    counts = torch.bincount(flat_indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def _macro_f1(confusion: torch.Tensor) -> float:
    true_positives = confusion.diagonal().float()
    predicted_totals = confusion.sum(dim=0).float()
    actual_totals = confusion.sum(dim=1).float()

    precision = torch.where(
        predicted_totals > 0, true_positives / predicted_totals, torch.zeros_like(true_positives)
    )
    recall = torch.where(
        actual_totals > 0, true_positives / actual_totals, torch.zeros_like(true_positives)
    )
    f1_per_class = torch.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    return f1_per_class.mean().item()


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
