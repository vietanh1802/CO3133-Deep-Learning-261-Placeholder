"""Image preprocessing and augmentation factories."""

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform

# Grayscale normalization: map input distribution roughly to [-1, 1].
_GRAYSCALE_MEAN = [0.5]
_GRAYSCALE_STD = [0.5]
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def build_train_transform() -> Transform:
    """Build preprocessing and training-only augmentation."""
    return v2.Compose(
        [
            # Convert input to a torchvision image tensor.
            v2.ToImage(),
            # Convert pixels to float32 and scale [0, 255] -> [0.0, 1.0].
            v2.ToDtype(torch.float32, scale=True),
            # Apply random geometric augmentation to improve generalization.
            v2.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),
            # Normalize using the same statistics expected by the model.
            v2.Normalize(mean=_GRAYSCALE_MEAN, std=_GRAYSCALE_STD),
        ]
    )


def build_eval_transform() -> Transform:
    """Build deterministic validation/test preprocessing."""
    return v2.Compose(
        [
            v2.ToImage(),
            # Keep evaluation preprocessing deterministic; no augmentation.
            v2.ToDtype(torch.float32, scale=True),
            # Use the same normalization as training for distribution consistency.
            v2.Normalize(mean=_GRAYSCALE_MEAN, std=_GRAYSCALE_STD),
        ]
    )


def build_cifar10_train_transform() -> Transform:
    """Build CIFAR-10 preprocessing with standard training augmentation."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
        ]
    )


def build_cifar10_eval_transform() -> Transform:
    """Build deterministic CIFAR-10 validation/test preprocessing."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
        ]
    )
