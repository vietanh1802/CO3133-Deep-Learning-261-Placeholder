"""Single-epoch training and validation boundaries."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


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
    """Train for one epoch and return sample-weighted aggregates."""
    raise NotImplementedError


def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> EpochResult:
    """Evaluate one epoch without updating model parameters."""
    raise NotImplementedError
