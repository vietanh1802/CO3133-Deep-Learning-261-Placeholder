"""Fashion-MNIST dataset construction."""

from pathlib import Path

from src.data.base import DataLoaders, ImageClassificationBatch, register_dataset


@register_dataset("fashion_mnist")
def build_fashion_mnist_loaders(
    data_dir: str | Path,
    *,
    batch_size: int,
    val_fraction: float,
    seed: int,
    num_workers: int = 0,
) -> DataLoaders[ImageClassificationBatch]:
    """Build the shared Fashion-MNIST train/validation/test split."""
    raise NotImplementedError


def class_names() -> tuple[str, ...]:
    """Return Fashion-MNIST class names in label-index order."""
    raise NotImplementedError
