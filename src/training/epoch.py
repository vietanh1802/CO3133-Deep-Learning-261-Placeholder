"""Shared optimizer updates, gradient accumulation, and distributed progress."""

import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils.distributed import (
    is_primary,
    world_size,
)
from src.utils.logging import ExperimentState


@dataclass(frozen=True)
class TrainingEpochResult:
    """Sample-weighted loss and optional additive task metric."""

    loss: float
    sample_count: int
    metric_total: float
    duration_seconds: float


def model_logits(model: nn.Module, images: Tensor) -> Tensor:
    """Read tensor logits from a model or a torchvision output mapping."""
    output = model(images)
    return output["out"] if isinstance(output, dict) else output


def run_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    scheduler: LRScheduler | None = None,
    accumulation_steps: int = 1,
    batch_metric: Callable[[Tensor, Tensor], float] | None = None,
) -> TrainingEpochResult:
    """Train once, synchronizing accumulated gradients before each optimizer update."""
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    num_batches = len(dataloader)
    loss_total = 0.0
    sample_total = 0
    metric_total = 0.0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    scheduled_samples = len(dataloader.sampler)  # ty: ignore[invalid-argument-type]
    if dataloader.drop_last:
        assert dataloader.batch_size is not None
        scheduled_samples = num_batches * dataloader.batch_size
    totals_device = device if world_size() > 1 else torch.device("cpu")
    totals = torch.zeros(3, dtype=torch.float64, device=totals_device)
    with tqdm(
        total=scheduled_samples * world_size(),
        desc=ExperimentState.TRAINING,
        unit="image",
        leave=False,
        disable=not is_primary(),
    ) as progress:
        for index, (images, targets) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            group_size = min(accumulation_steps, num_batches - index + index % accumulation_steps)
            update = (index + 1) % accumulation_steps == 0 or index + 1 == num_batches
            context = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and not update
                else nullcontext()
            )
            with context:
                logits = model_logits(model, images)
                loss = loss_fn(logits, targets)
                (loss / group_size).backward()
            loss_total += loss.item() * len(targets)
            sample_total += len(targets)
            if batch_metric is not None:
                metric_total += batch_metric(logits.detach(), targets)
            if update:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                totals = torch.tensor(
                    [loss_total, sample_total, metric_total],
                    dtype=torch.float64,
                    device=totals_device,
                )
                if world_size() > 1:
                    dist.all_reduce(totals)
                if is_primary():
                    progress.update(int(totals[1].item()) - progress.n)
                    progress.set_postfix(loss=f"{(totals[0] / totals[1]).item():.4f}")

    if sample_total == 0:
        raise ValueError("dataloader yielded no samples")
    return TrainingEpochResult(
        loss=(totals[0] / totals[1]).item(),
        sample_count=int(totals[1].item()),
        metric_total=totals[2].item(),
        duration_seconds=time.perf_counter() - started,
    )
