"""CIFAR-10 dataset construction."""

from pathlib import Path

from torchvision.datasets import CIFAR10

from src.data.base import (
    DataLoaders,
    ImageClassificationBatch,
    build_image_classification_loaders,
    register_dataset,
)
from src.data.transform import build_cifar10_eval_transform, build_cifar10_train_transform

CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@register_dataset("cifar10")
def build_cifar10_loaders(
    data_dir: str | Path,
    *,
    batch_size: int,
    val_fraction: float,
    seed: int,
    num_workers: int = 0,
) -> DataLoaders[ImageClassificationBatch]:
    """Build CIFAR-10 loaders using a reproducible train/validation split."""
    return build_image_classification_loaders(
        train_dataset=CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=build_cifar10_train_transform(),
        ),
        validation_dataset=CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=build_cifar10_eval_transform(),
        ),
        test_dataset=CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=build_cifar10_eval_transform(),
        ),
        batch_size=batch_size,
        val_fraction=val_fraction,
        seed=seed,
        num_workers=num_workers,
    )
