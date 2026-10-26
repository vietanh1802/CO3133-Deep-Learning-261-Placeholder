"""Inference-cost measurement for the compute-cost analysis."""

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class InferenceCost:
    """Forward-pass time of one model over one split, excluding data loading."""

    images: int
    batch_size: int
    seconds: float
    milliseconds_per_image: float
    images_per_second: float
    peak_memory_mb: float | None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    warmup_batches: int = 2,
) -> InferenceCost:
    """Time every batch's forward pass after ``warmup_batches`` untimed passes.

    Data loading and host-to-device copies are outside the timed region, so the
    figure is the model's cost alone. Peak memory is reported on CUDA only.
    """
    if warmup_batches < 0:
        raise ValueError("warmup_batches must be non-negative")
    model.to(device).eval()
    first = next(iter(dataloader), None)
    if first is None:
        raise ValueError("dataloader yielded no samples")
    warmup_images = first[0].to(device)
    for _ in range(warmup_batches):
        model(warmup_images)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    seconds = 0.0
    count = 0
    for images, _ in dataloader:
        images = images.to(device)
        _synchronize(device)
        started = time.perf_counter()
        model(images)
        _synchronize(device)
        seconds += time.perf_counter() - started
        count += len(images)

    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    return InferenceCost(
        images=count,
        batch_size=len(first[0]),
        seconds=seconds,
        milliseconds_per_image=1000 * seconds / count,
        images_per_second=count / seconds,
        peak_memory_mb=peak,
    )
