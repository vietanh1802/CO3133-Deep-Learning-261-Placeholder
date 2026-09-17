"""Held-out test evaluation boundary."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import ClassificationMetrics


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and evidence collected from one held-out evaluation."""

    metrics: ClassificationMetrics
    sample_indices: torch.Tensor
    targets: torch.Tensor
    predictions: torch.Tensor
    probabilities: torch.Tensor
    parameter_count: int
    inference_seconds: float


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> EvaluationResult:
    """Evaluate one selected checkpoint on the held-out test split."""
    raise NotImplementedError
