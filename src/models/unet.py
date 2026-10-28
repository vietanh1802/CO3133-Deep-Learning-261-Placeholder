"""Semantic segmentation models."""

import torch
from torch import nn

from src.models.base import Model, register_model


def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


@register_model("unet")
class UNet(Model):
    """Four-level U-Net trained from scratch, after Ronneberger et al. (2015)."""

    depth = 4

    def __init__(self, in_channels: int = 3, num_classes: int = 8, base_width: int = 32) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes <= 0 or base_width <= 0:
            raise ValueError("in_channels, num_classes and base_width must be positive")

        widths = [base_width * 2**level for level in range(self.depth + 1)]
        self.encoders = nn.ModuleList([_double_conv(in_channels, widths[0])])
        self.encoders.extend(
            _double_conv(widths[level], widths[level + 1]) for level in range(self.depth)
        )
        self.pool = nn.MaxPool2d(2)
        self.upsamplers = nn.ModuleList(
            nn.ConvTranspose2d(widths[level + 1], widths[level], 2, stride=2)
            for level in reversed(range(self.depth))
        )
        self.decoders = nn.ModuleList(
            _double_conv(widths[level] * 2, widths[level]) for level in reversed(range(self.depth))
        )
        self.head = nn.Conv2d(widths[0], num_classes, 1)
        self.output_dim = num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-pixel logits shaped ``[batch, num_classes, height, width]``."""
        if images.shape[-2] % 2**self.depth or images.shape[-1] % 2**self.depth:
            raise ValueError(f"image height and width must be divisible by {2**self.depth}")

        skips = []
        features = images
        for level, encoder in enumerate(self.encoders):
            features = encoder(features if level == 0 else self.pool(features))
            skips.append(features)
        skips.pop()
        for upsample, decoder in zip(self.upsamplers, self.decoders, strict=True):
            features = decoder(torch.cat((skips.pop(), upsample(features)), dim=1))
        return self.head(features)
