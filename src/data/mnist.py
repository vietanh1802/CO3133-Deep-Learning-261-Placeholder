"""MNIST loaders for development and debugging only."""

from pathlib import Path

from src.data.base import DataLoaders, ImageClassificationBatch, register_dataset


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
    raise NotImplementedError
