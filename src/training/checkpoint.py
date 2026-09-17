"""Checkpoint persistence boundary."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from src.utils.seed import capture_rng_state, restore_rng_state


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
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": asdict(metadata),
        "rng_state": capture_rng_state(),
    }

    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> CheckpointMetadata:
    """Restore a checkpoint and return its metadata."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    required_keys = {"model", "optimizer", "metadata", "rng_state"}
    missing_keys = required_keys - set(payload)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"checkpoint is missing keys: {missing}")

    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])

    return CheckpointMetadata(**payload["metadata"])
