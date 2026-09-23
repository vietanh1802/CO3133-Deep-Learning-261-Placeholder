"""Fashion-MNIST dataset construction."""

from pathlib import Path

from torchvision.datasets import FashionMNIST

from src.data.base import (
    DataLoaders,
    ImageClassificationBatch,
    build_image_classification_loaders,
    register_dataset,
)
from src.data.transform import build_eval_transform, build_train_transform

FASHION_MNIST_CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


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
    return build_image_classification_loaders(
        train_dataset=FashionMNIST(
            root=data_dir, train=True, download=True, transform=build_train_transform()
        ),
        validation_dataset=FashionMNIST(
            root=data_dir, train=True, download=True, transform=build_eval_transform()
        ),
        test_dataset=FashionMNIST(
            root=data_dir, train=False, download=True, transform=build_eval_transform()
        ),
        batch_size=batch_size,
        val_fraction=val_fraction,
        seed=seed,
        num_workers=num_workers,
    )


def class_names() -> tuple[str, ...]:
    """Return Fashion-MNIST class names in label-index order."""
    return FASHION_MNIST_CLASS_NAMES
