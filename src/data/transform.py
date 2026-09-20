"""Image preprocessing and augmentation factories."""

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform

# Grayscale normalization: map input distribution roughly to [-1, 1].
_NORMALIZE_MEAN = [0.5]
_NORMALIZE_STD = [0.5]


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
            v2.Normalize(mean=_NORMALIZE_MEAN, std=_NORMALIZE_STD),
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
            v2.Normalize(mean=_NORMALIZE_MEAN, std=_NORMALIZE_STD),
        ]
    )
