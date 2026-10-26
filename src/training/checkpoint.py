"""Shared checkpoint persistence and classification metadata."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.utils.distributed import (
    is_primary,
    rank,
    unwrap,
    world_size,
)
from src.utils.seed import capture_rng_state, restore_rng_state


@dataclass(frozen=True)
class CheckpointMetadata:
    """Information needed to identify and reproduce a saved training state."""

    epoch: int
    global_step: int
    best_validation_loss: float
    config: dict[str, Any]


def save_state(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    metadata: Mapping[str, Any],
    *,
    scheduler: LRScheduler | None = None,
) -> None:
    """Collect RNG states on every rank, then atomically save on rank 0."""
    local_rng = capture_rng_state()
    rng_states: list[Any] = [local_rng]
    if world_size() > 1:
        rng_states = [None] * world_size()
        dist.all_gather_object(rng_states, local_rng)
    if not is_primary():
        return
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    payload = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": dict(metadata),
        "rng_state": local_rng,
        "rng_states": rng_states,
        "world_size": world_size(),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_state(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    *,
    scheduler: LRScheduler | None = None,
    require_scheduler: bool = False,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a shared payload, including older single-process RNG checkpoints."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    required_keys = {"model", "optimizer", "metadata", "rng_state"}
    if require_scheduler or scheduler is not None:
        required_keys.add("scheduler")
    missing_keys = required_keys - set(payload)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"checkpoint is missing keys: {missing}")
    if restore_rng and payload.get("world_size", 1) != world_size():
        raise ValueError("resume requires the same world size as the saved checkpoint")
    unwrap(model).load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        state = payload.get("rng_states", [payload["rng_state"]])[rank()]
        state["torch"] = state["torch"].cpu()
        if "cuda" in state:
            state["cuda"] = [value.cpu() for value in state["cuda"]]
        restore_rng_state(state)
    return payload["metadata"]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    metadata: CheckpointMetadata,
) -> None:
    """Persist the classification training state."""
    save_state(path, model, optimizer, asdict(metadata))


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> CheckpointMetadata:
    """Restore a checkpoint and return its classification metadata."""
    return CheckpointMetadata(
        **load_state(path, model, optimizer, map_location=map_location, restore_rng=restore_rng)
    )
