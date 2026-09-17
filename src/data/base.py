"""Shared dataset contracts and construction boundary."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch import Tensor
from torch.utils.data import DataLoader

BatchT = TypeVar("BatchT")
ImageClassificationBatch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class DataLoaders(Generic[BatchT]):
    """Train, validation, and test loaders for one task-specific batch type."""

    train: DataLoader[BatchT]
    validation: DataLoader[BatchT]
    test: DataLoader[BatchT]


DatasetBuilder = Callable[..., DataLoaders[ImageClassificationBatch]]
DatasetBuilderT = TypeVar("DatasetBuilderT", bound=DatasetBuilder)
_DATASET_REGISTRY: dict[str, DatasetBuilder] = {}


def register_dataset(name: str) -> Callable[[DatasetBuilderT], DatasetBuilderT]:
    """Register an image-classification loader builder under a factory name."""
    key = name.strip().lower().replace("-", "_")
    if not key:
        raise ValueError("Dataset name cannot be empty")

    def decorator(builder: DatasetBuilderT) -> DatasetBuilderT:
        if key in _DATASET_REGISTRY:
            raise ValueError(f"Dataset {key!r} is already registered")
        _DATASET_REGISTRY[key] = builder
        return builder

    return decorator


def build_dataloaders(
    name: str,
    config: Mapping[str, Any],
) -> DataLoaders[ImageClassificationBatch]:
    """Construct dataset from its name and dataset-specific settings."""
    key = name.strip().lower().replace("-", "_")
    if key not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {available}")
    return _DATASET_REGISTRY[key](**dict(config))
