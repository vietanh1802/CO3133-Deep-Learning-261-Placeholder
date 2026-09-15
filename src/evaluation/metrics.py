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
    raise NotImplementedError


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    raise NotImplementedError
