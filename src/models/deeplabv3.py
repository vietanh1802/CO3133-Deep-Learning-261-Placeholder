"""Pretrained DeepLabV3-ResNet50 adapted to dense SUIM labels."""

from typing import Self

import torch
from torch import nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

from src.models.base import Model, register_model


@register_model("deeplabv3")
class DeepLabV3(Model):
    """Return full-resolution logits while optionally fixing BatchNorm statistics.

    Freezing BatchNorm statistics supports small batches, including a final batch
    of one image. Its affine parameters and all other model parameters still train.
    """

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = True,
        freeze_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
        self.network = deeplabv3_resnet50(weights=weights, weights_backbone=None)
        self.network.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.network.aux_classifier = None
        self.output_dim = num_classes
        self.freeze_batch_norm = freeze_batch_norm
        self.train()

    def train(self, mode: bool = True) -> Self:
        """Set training mode, keeping BatchNorm running statistics fixed if requested."""
        super().train(mode)
        if mode and self.freeze_batch_norm:
            for module in self.network.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map float RGB ``[B, 3, H, W]`` images to ``[B, num_classes, H, W]`` logits."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        return self.network(images)["out"]
