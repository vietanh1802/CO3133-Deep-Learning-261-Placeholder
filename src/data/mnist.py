"""MNIST loaders for development and debugging only."""

from pathlib import Path

from torchvision.datasets import MNIST

from src.data.base import (
    DataLoaders,
    ImageClassificationBatch,
    build_image_classification_loaders,
    register_dataset,
)
from src.data.transform import build_eval_transform, build_train_transform


@register_dataset("mnist")
def build_mnist_loaders(
    data_dir: str | Path,
    *,
    batch_size: int,
    val_fraction: float,
    seed: int,
    num_workers: int = 0,
) -> DataLoaders[ImageClassificationBatch]:
    """Build MNIST loaders using a reproducible split."""
    return build_image_classification_loaders(
        train_dataset=MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=build_train_transform(),
        ),
        validation_dataset=MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=build_eval_transform(),
        ),
        test_dataset=MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=build_eval_transform(),
        ),
        batch_size=batch_size,
        val_fraction=val_fraction,
        seed=seed,
        num_workers=num_workers,
    )
