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
    if targets.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if predictions.ndim != 1:
        raise ValueError("predictions must be one-dimensional")
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if targets.numel() == 0:
        raise ValueError("targets must not be empty")

    target_values = targets.to(torch.long)
    prediction_values = predictions.to(torch.long)
    for name, values in (("targets", target_values), ("predictions", prediction_values)):
        if values.numel() and (int(values.min()) < 0 or int(values.max()) >= num_classes):
            raise ValueError(f"{name} must contain labels in [0, {num_classes})")

    flat_index = target_values * num_classes + prediction_values
    confusion_matrix = torch.bincount(
        flat_index,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    total = int(confusion_matrix.sum())
    correct = int(confusion_matrix.diagonal().sum())
    accuracy = correct / total

    true_positives = confusion_matrix.diagonal().to(torch.float64)
    predicted_per_class = confusion_matrix.sum(dim=0).to(torch.float64)
    actual_per_class = confusion_matrix.sum(dim=1).to(torch.float64)

    denominator = predicted_per_class + actual_per_class
    per_class_f1 = torch.where(
        denominator > 0,
        2.0 * true_positives / denominator,
        torch.zeros_like(denominator),
    )
    macro_f1 = float(per_class_f1.mean())

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        confusion_matrix=confusion_matrix,
    )


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
