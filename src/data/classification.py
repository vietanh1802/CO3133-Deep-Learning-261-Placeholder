"""Shared train-validation splitting and loaders for image classification."""

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.base import DataLoaders, ImageBatch
from src.utils.seed import make_generator, seed_worker


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
) -> DataLoaders[ImageBatch]:
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
