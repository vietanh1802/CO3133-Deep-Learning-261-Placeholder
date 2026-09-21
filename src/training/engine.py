"""Single-epoch training and validation boundaries."""

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.logging import ExperimentState, progress_bar


@dataclass(frozen=True)
class EpochResult:
    """Sample-weighted aggregates returned by one complete epoch."""

    loss: float
    accuracy: float
    sample_count: int
    duration_seconds: float


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> EpochResult:
    """Train for one epoch and return sample-weighted aggregates.

    ``loss_fn`` must reduce a batch to its mean, which is the default for
    :class:`~torch.nn.CrossEntropyLoss`. The caller is responsible for moving
    ``model`` onto ``device`` before the first epoch.
    """
    model.train()
    batches = progress_bar(dataloader, desc=ExperimentState.TRAINING, leave=False)

    loss_total = 0.0
    correct_total = 0
    sample_total = 0
    start = time.perf_counter()

    for images, targets in batches:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        loss_total += loss.item() * batch_size
        correct_total += int((logits.argmax(dim=1) == targets).sum().item())
        sample_total += batch_size

    return _aggregate(loss_total, correct_total, sample_total, time.perf_counter() - start)


def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> EpochResult:
    """Evaluate one epoch without updating model parameters."""
    model.eval()
    batches = progress_bar(dataloader, desc=ExperimentState.VALIDATING, leave=False)

    loss_total = 0.0
    correct_total = 0
    sample_total = 0
    start = time.perf_counter()

    with torch.no_grad():
        for images, targets in batches:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, targets)

            batch_size = targets.shape[0]
            loss_total += loss.item() * batch_size
            correct_total += int((logits.argmax(dim=1) == targets).sum().item())
            sample_total += batch_size

    return _aggregate(loss_total, correct_total, sample_total, time.perf_counter() - start)


def _aggregate(
    loss_total: float,
    correct_total: int,
    sample_total: int,
    duration_seconds: float,
) -> EpochResult:
    """Divide running totals by the exact number of samples seen."""
    if sample_total == 0:
        raise ValueError("dataloader yielded no samples")

    return EpochResult(
        loss=loss_total / sample_total,
        accuracy=correct_total / sample_total,
        sample_count=sample_total,
        duration_seconds=duration_seconds,
    )
