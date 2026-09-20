"""Dataset contracts and construction boundary."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.seed import make_generator, seed_worker

BatchT = TypeVar("BatchT")
ImageClassificationBatch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class DataLoaders(Generic[BatchT]):
    """Container for train, validation, and test data loaders."""

    train: DataLoader[BatchT]
    validation: DataLoader[BatchT]
    test: DataLoader[BatchT]


DatasetBuilder = Callable[..., DataLoaders[ImageClassificationBatch]]
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
) -> DataLoaders[ImageClassificationBatch]:
    """Build task-specific data loaders from a registered dataset name."""
    key = name.strip().lower().replace("-", "_")

    if key not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY))
        raise ValueError(
            f"Unknown dataset {name!r}. Available datasets: {available}"
        )

    return _DATASET_REGISTRY[key](**dict(config))


def split_train_validation_indices(
    num_samples: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split sample indices into reproducible, disjoint train/validation sets."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0.0, 1.0)")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(
        num_samples,
        generator=generator,
    ).tolist()

    num_val = int(num_samples * val_fraction)

    return permutation[num_val:], permutation[:num_val]

def build_image_classification_loaders(
    *,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    val_fraction: float,
    seed: int,
    num_workers: int,
) -> DataLoaders[ImageClassificationBatch]:
    """Split off a validation set and wrap all three datasets in DataLoaders."""
    train_indices, val_indices = split_train_validation_indices(
        len(train_dataset),  # ty: ignore[invalid-argument-type]
        val_fraction,
        seed,
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    return DataLoaders(train=train_loader, validation=validation_loader, test=test_loader)
