"""Task-specific image-classification model."""

import torch

from src.models.base import Model


class ImageClassifier(Model):
    """Attach a classification head to a reusable image encoder."""

    def __init__(self, encoder: Model, num_classes: int = 10) -> None:
        super().__init__()
        pass

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # ty: ignore[empty-body]
        """Return unnormalized logits shaped ``[batch, num_classes]``."""
        pass
