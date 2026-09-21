"""Held-out test evaluation boundary."""

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    count_trainable_parameters,
)


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


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> EvaluationResult:
    """Evaluate one selected checkpoint on the held-out test split."""
    model.eval()
    model.to(device)

    batch_indices: list[torch.Tensor] = []
    batch_targets: list[torch.Tensor] = []
    batch_probabilities: list[torch.Tensor] = []

    seen = 0
    inference_seconds = 0.0

    for batch in dataloader:
        images, targets = batch[0], batch[1]
        images = images.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started

        probabilities = torch.softmax(logits.to(torch.float32), dim=1)

        batch_probabilities.append(probabilities.cpu())
        batch_targets.append(targets.detach().cpu().to(torch.long).flatten())
        batch_indices.append(torch.arange(seen, seen + images.shape[0]))
        seen += images.shape[0]

    if seen == 0:
        raise ValueError("dataloader produced no samples")

    sample_indices = torch.cat(batch_indices)
    targets = torch.cat(batch_targets)
    probabilities = torch.cat(batch_probabilities)
    predictions = probabilities.argmax(dim=1)

    num_classes = probabilities.shape[1]
    metrics = compute_classification_metrics(
        targets,
        predictions,
        num_classes=num_classes,
    )

    return EvaluationResult(
        metrics=metrics,
        sample_indices=sample_indices,
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        parameter_count=count_trainable_parameters(model),
        inference_seconds=inference_seconds,
    )
