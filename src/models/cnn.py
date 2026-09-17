"""Convolutional image encoder."""

import torch

from src.models.base import Model, register_model


@register_model("cnn")
class CNN(Model):
    """Encode 28-by-28 grayscale images with a self-designed CNN."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        pass

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # ty: ignore[empty-body]
        """Return encoded image features."""
        pass
