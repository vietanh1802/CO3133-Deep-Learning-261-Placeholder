"""Training and validation engines."""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from src.training.checkpoint import CheckpointMetadata, save_checkpoint
from src.utils.logging import ExperimentState, progress_bar


@dataclass(frozen=True)
class EpochResult:
    """Sample-weighted aggregates returned by one complete epoch."""

    loss: float
    accuracy: float
    sample_count: int
    duration_seconds: float


@dataclass(frozen=True)
class FitResult:
    """Training history and best-checkpoint information."""

    training: tuple[EpochResult, ...]
    validation: tuple[EpochResult, ...]
    best_epoch: int
    best_validation_loss: float
    global_step: int
    duration_seconds: float


def fit(
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    epochs: int,
    checkpoint_path: str | Path,
    checkpoint_config: Mapping[str, Any],
) -> FitResult:
    """Train and validate for several epochs, saving the best checkpoint."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    model.to(device)
    training_history: list[EpochResult] = []
    validation_history: list[EpochResult] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(model, train_dataloader, optimizer, loss_fn, device)
        validation_result = eval_one_epoch(model, validation_dataloader, loss_fn, device)
        global_step += len(train_dataloader)
        training_history.append(train_result)
        validation_history.append(validation_result)

        logger.info(
            "Epoch {}/{} | train loss {:.4f}, accuracy {:.4f} | "
            "validation loss {:.4f}, accuracy {:.4f}",
            epoch,
            epochs,
            train_result.loss,
            train_result.accuracy,
            validation_result.loss,
            validation_result.accuracy,
        )

        if validation_result.loss < best_validation_loss:
            best_validation_loss = validation_result.loss
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                CheckpointMetadata(
                    epoch=epoch,
                    global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    config=dict(checkpoint_config),
                ),
            )

    return FitResult(
        training=tuple(training_history),
        validation=tuple(validation_history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        global_step=global_step,
        duration_seconds=time.perf_counter() - started,
    )


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
