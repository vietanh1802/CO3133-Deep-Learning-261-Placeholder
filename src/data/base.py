"""Shared dataset contracts and construction boundary."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch.utils.data import DataLoader

Batch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class DataLoaders:
    """The three loaders shared by every model in the main comparison."""

    train: DataLoader[Batch]
    validation: DataLoader[Batch]
    test: DataLoader[Batch]


def build_dataloaders(config: Mapping[str, Any]) -> DataLoaders:
    """Dispatch to the configured dataset builder."""
    raise NotImplementedError
