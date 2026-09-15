"""Transformer image classifier."""

import torch

from src.models.base import ImageClassifier


class TransformerClassifier(ImageClassifier):
    """Classify an image represented as row, column, or patch tokens."""

    def __init__(
        self,
        token_dim: int = 28,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits."""
        raise NotImplementedError
