"""Random-number-generator controls for reproducible experiments."""

from typing import Any


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and DataLoader workers."""
    raise NotImplementedError


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG state for exact checkpoint resumption."""
    raise NotImplementedError


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state returned by :func:`capture_rng_state`."""
    raise NotImplementedError
