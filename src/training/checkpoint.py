"""Checkpoint persistence boundary."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

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
    run_name: str | None = None,
) -> CheckpointMetadata:
    """Restore a checkpoint and return its metadata."""
    checkpoint_path = Path(path)
    if checkpoint_path.suffix.lower() == ".zip":
        if run_name is None:
            raise ValueError("run_name is required when loading an assignment ZIP")
        with ZipFile(checkpoint_path) as archive:
            candidates = (f"checkpoints/{run_name}/best.pt", f"{run_name}/best.pt")
            members = [name for name in candidates if name in archive.namelist()]
            if len(members) != 1:
                raise FileNotFoundError(
                    f"expected one best.pt for {run_name!r} in {checkpoint_path}; found {members}"
                )
            data = BytesIO(archive.read(members[0]))
        payload = torch.load(data, map_location=map_location, weights_only=True)
    else:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
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


def archive_checkpoints(archive_path: Path, checkpoints: Mapping[str, Path]) -> None:
    """Atomically update selected runs in an assignment ZIP, preserving other runs."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    replacements = {f"checkpoints/{name}/best.pt": path for name, path in checkpoints.items()}
    replaced_members = set(replacements) | {f"{name}/best.pt" for name in checkpoints}
    temporary_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    try:
        with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED) as output:
            if archive_path.exists():
                with ZipFile(archive_path) as previous:
                    for member in previous.infolist():
                        if (
                            member.filename not in replaced_members
                            and member.filename != "manifest.json"
                        ):
                            output.writestr(member, previous.read(member))
            for member, path in replacements.items():
                output.write(path, arcname=member)
        temporary_path.replace(archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)
