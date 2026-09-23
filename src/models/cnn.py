"""Convolutional image encoder."""

import torch
from torch import nn

from src.models.base import Model, register_model


@register_model("cnn")
class CNN(Model):
    """Encode 28-by-28 grayscale images with a self-designed CNN."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.in_channels = in_channels
        channels = (32, 64)
        self.output_dim = channels[-1]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return encoded image features."""
        if images.dim() != 4:
            raise ValueError(f"expected a 4D image batch, got {images.dim()} dimensions")
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels per image, got {images.shape[1]}"
            )
        feature_maps = self.features(images)
        return self.flatten(self.pool(feature_maps))
