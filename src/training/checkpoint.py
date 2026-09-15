"""Checkpoint persistence boundary."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class CheckpointMetadata:
    """Information needed to identify and reproduce a saved training state."""

    epoch: int
    global_step: int
    best_validation_loss: float
    config: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    metadata: CheckpointMetadata,
) -> None:
    """Persist model, optimizer, metadata, and random-number-generator state."""
    raise NotImplementedError


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> CheckpointMetadata:
    """Restore a checkpoint and return its metadata."""
    raise NotImplementedError
