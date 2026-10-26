"""Dataset contracts and construction boundary."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch import Tensor
from torch.utils.data import DataLoader

BatchT = TypeVar("BatchT")
ImageBatch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class DataLoaders(Generic[BatchT]):
    """Container for train, validation, and test data loaders."""

    train: DataLoader[BatchT]
    validation: DataLoader[BatchT]
    test: DataLoader[BatchT]


DatasetBuilder = Callable[..., DataLoaders[ImageBatch]]
DatasetBuilderT = TypeVar("DatasetBuilderT", bound=DatasetBuilder)

_DATASET_REGISTRY: dict[str, DatasetBuilder] = {}


def register_dataset(name: str) -> Callable[[DatasetBuilderT], DatasetBuilderT]:
    """Register a dataset builder under a normalized factory name."""
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
) -> DataLoaders[ImageBatch]:
    """Build task-specific data loaders from a registered dataset name."""
    key = name.strip().lower().replace("-", "_")

    if key not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {available}")

    return _DATASET_REGISTRY[key](**dict(config))
