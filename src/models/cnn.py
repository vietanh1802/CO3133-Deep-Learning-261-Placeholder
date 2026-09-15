"""Convolutional image classifier."""

import torch

from src.models.base import ImageClassifier


class CNNClassifier(ImageClassifier):
    """Self-designed CNN for 28-by-28 grayscale images."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits."""
        raise NotImplementedError
