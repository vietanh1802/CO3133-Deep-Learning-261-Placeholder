"""Dataset and preprocessing interfaces."""

from src.data.base import (
    DataLoaders,
    ImageClassificationBatch,
    build_dataloaders,
    register_dataset,
)
from src.data.cifar10 import CIFAR10_CLASS_NAMES, build_cifar10_loaders
from src.data.fashion_mnist import FASHION_MNIST_CLASS_NAMES, build_fashion_mnist_loaders
from src.data.mnist import build_mnist_loaders


def class_names(name: str) -> tuple[str, ...]:
    """Return class names in label-index order for a built-in dataset."""
    key = name.strip().lower().replace("-", "_")
    if key == "fashion_mnist":
        return FASHION_MNIST_CLASS_NAMES
    if key == "mnist":
        return tuple(str(index) for index in range(10))
    if key == "cifar10":
        return CIFAR10_CLASS_NAMES
    raise ValueError(f"Class names are unavailable for dataset {name!r}")


__all__ = [
    "DataLoaders",
    "ImageClassificationBatch",
    "build_cifar10_loaders",
    "build_dataloaders",
    "build_fashion_mnist_loaders",
    "build_mnist_loaders",
    "class_names",
    "register_dataset",
]
