"""Task-specific image-classification model."""

import torch
from torch import nn

from src.models.base import Model


class ImageClassifier(Model):
    """Attach a classification head to a reusable image encoder."""

    def __init__(self, encoder: Model, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if encoder.output_dim <= 0:
            raise ValueError("encoder.output_dim must be positive")

        self.encoder = encoder
        self.output_dim = num_classes
        self.classifier = nn.Linear(encoder.output_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return unnormalized logits shaped ``[batch, num_classes]``."""
        return self.classifier(self.encoder(images))
