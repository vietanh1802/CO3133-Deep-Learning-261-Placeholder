"""Convolutional image encoder."""

import torch
from torch import nn

from src.models.base import Model, register_model


@register_model("cnn")
class CNN(Model):
    """Encode images while preserving the spatial layout of convolutional features."""

    def __init__(self, in_channels: int = 1, image_size: int = 28) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if image_size < 4:
            raise ValueError("image_size must be at least 4")
        self.image_size = image_size
        self.in_channels = in_channels
        channels = (32, 64)
        self.output_dim = 128
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
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * (image_size // 4) ** 2, self.output_dim),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return encoded image features."""
        if images.dim() != 4:
            raise ValueError(f"expected a 4D image batch, got {images.dim()} dimensions")
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels per image, got {images.shape[1]}"
            )
        if images.shape[2:] != (self.image_size, self.image_size):
            raise ValueError(f"expected {self.image_size}-by-{self.image_size} images")
        feature_maps = self.features(images)
        return self.projection(feature_maps)
