"""Segmentation training with gradient accumulation, mIoU selection, and resume."""

import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader, DistributedSampler

from src.evaluation.segmentation import (
    SegmentationMetrics,
    confusion_matrix,
    metrics_from_confusion,
)
from src.training.checkpoint import load_state, save_state
from src.training.epoch import model_logits, run_training_epoch
from src.utils.distributed import (
    is_primary,
    rank,
    unwrap,
    world_size,
)
from src.utils.logging import ExperimentState, progress_bar


@dataclass(frozen=True)
class EpochRecord:
    """One epoch of training loss and validation scores."""

    epoch: int
    train_loss: float
    validation_loss: float
    validation_miou: float
    validation_mdice: float
    learning_rate: float
    seconds: float


@dataclass(frozen=True)
class SegmentationFitResult:
    """Full history, including epochs completed before a resume."""

    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_miou: float
    duration_seconds: float


@dataclass(frozen=True)
class SegmentationCheckpointMetadata:
    """Information needed to identify and resume a saved segmentation training state."""

    epoch: int
    global_step: int
    best_validation_miou: float
    config: dict[str, Any]


def save_training_state(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    metadata: SegmentationCheckpointMetadata,
) -> None:
    """Collect RNG states on all ranks, then let rank 0 persist the training state."""
    save_state(path, model, optimizer, asdict(metadata), scheduler=scheduler)


def load_training_state(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    *,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> SegmentationCheckpointMetadata:
    """Restore a segmentation checkpoint and return its metadata."""
    return SegmentationCheckpointMetadata(
        **load_state(
            path,
            model,
            optimizer,
            scheduler=scheduler,
            require_scheduler=True,
            map_location=map_location,
            restore_rng=restore_rng,
        )
    )


def optimizer_steps_per_epoch(num_batches: int, accumulation_steps: int) -> int:
    """Count optimizer steps when a final partial accumulation group also steps."""
    return math.ceil(num_batches / accumulation_steps)


def warmup_cosine(optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warm-up to the base rate, then cosine decay to zero, per optimizer step."""
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in the range [0, total_steps)")

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, SegmentationMetrics]:
    """Return the mean validation loss and metrics from pooled confusion counts."""
    model.eval()
    counts = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    loss_total = 0.0
    sample_total = 0

    for images, targets in progress_bar(dataloader, desc=ExperimentState.VALIDATING, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model_logits(model, images)
        loss_total += loss_fn(logits, targets).item() * len(targets)
        sample_total += len(targets)
        counts += confusion_matrix(targets, logits.argmax(1), num_classes=num_classes)

    if sample_total == 0:
        raise ValueError("dataloader yielded no samples")
    return loss_total / sample_total, metrics_from_confusion(counts)


def fit_segmentation(
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    epochs: int,
    accumulation_steps: int,
    seed: int,
    checkpoint_dir: str | Path,
    run_dir: str | Path,
    checkpoint_config: Mapping[str, Any],
    num_classes: int = 8,
    resume: bool = False,
) -> SegmentationFitResult:
    """Train, keep the highest-validation-mIoU checkpoint, and save state every epoch.

    ``best.pt`` holds the selected model and ``last.pt`` the latest full training
    state; ``history.json`` in ``run_dir`` is rewritten after every epoch.
    """
    checkpoint_dir, run_dir = Path(checkpoint_dir), Path(run_dir)
    last_path, best_path = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
    history_path = run_dir / "history.json"
    if is_primary():
        run_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(model, DistributedDataParallel):
        model.to(device)

    history: list[EpochRecord] = []
    if resume:
        metadata = load_training_state(last_path, model, optimizer, scheduler, map_location=device)
        records = json.loads(history_path.read_text(encoding="utf-8"))
        # History is written before last.pt, so it can hold one epoch the checkpoint lacks.
        history = [EpochRecord(**record) for record in records[: metadata.epoch]]
        if is_primary():
            logger.info("Resuming after epoch {}", metadata.epoch)

    best = max(history, key=lambda record: record.validation_miou, default=None)
    steps_per_epoch = optimizer_steps_per_epoch(len(train_dataloader), accumulation_steps)

    for epoch in range(len(history) + 1, epochs + 1):
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        if train_dataloader.generator is not None:
            train_dataloader.generator.manual_seed(seed + epoch + rank())
        learning_rate = optimizer.param_groups[0]["lr"]
        started = time.perf_counter()
        train_loss = run_training_epoch(
            model,
            train_dataloader,
            optimizer,
            loss_fn,
            device,
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
        ).loss
        validation_result: list[Any] = [None]
        if is_primary():
            validation_result[0] = validate(
                unwrap(model), validation_dataloader, loss_fn, device, num_classes
            )
        if world_size() > 1:
            dist.broadcast_object_list(validation_result, src=0)
        validation_loss, metrics = validation_result[0]
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            validation_miou=metrics.mean_iou,
            validation_mdice=metrics.mean_dice,
            learning_rate=learning_rate,
            seconds=time.perf_counter() - started,
        )
        history.append(record)
        if is_primary():
            logger.info(
                "Epoch {}/{} | train loss {:.4f} | validation loss {:.4f}, mIoU {:.4f}, "
                "mDice {:.4f} | {:.0f}s",
                epoch,
                epochs,
                train_loss,
                validation_loss,
                metrics.mean_iou,
                metrics.mean_dice,
                record.seconds,
            )

        improved = best is None or record.validation_miou > best.validation_miou
        if improved:
            best = record
        metadata = SegmentationCheckpointMetadata(
            epoch=epoch,
            global_step=epoch * steps_per_epoch,
            best_validation_miou=best.validation_miou,
            config=dict(checkpoint_config),
        )
        if improved:
            save_training_state(best_path, model, optimizer, scheduler, metadata)
        if is_primary():
            history_path.write_text(
                json.dumps([asdict(item) for item in history], indent=2), encoding="utf-8"
            )
        save_training_state(last_path, model, optimizer, scheduler, metadata)

    if world_size() > 1:
        dist.barrier()
    if best is None:
        raise ValueError("no epochs were run")
    return SegmentationFitResult(
        history=tuple(history),
        best_epoch=best.epoch,
        best_validation_miou=best.validation_miou,
        duration_seconds=sum(record.seconds for record in history),
    )
