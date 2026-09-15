"""Linear image classifier."""

import torch

from src.models.base import ImageClassifier


class LinearClassifier(ImageClassifier):
    """Flatten an image and return unnormalized class logits."""

    def __init__(self, input_dim: int = 28 * 28, num_classes: int = 10) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``[batch, num_classes]``."""
        raise NotImplementedError
