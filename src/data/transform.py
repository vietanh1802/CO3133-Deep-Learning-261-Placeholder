"""Image preprocessing and augmentation factories."""

from torchvision.transforms.v2 import Transform


def build_train_transform() -> Transform:
    """Build preprocessing and training-only augmentation."""
    raise NotImplementedError


def build_eval_transform() -> Transform:
    """Build deterministic validation/test preprocessing."""
    raise NotImplementedError
