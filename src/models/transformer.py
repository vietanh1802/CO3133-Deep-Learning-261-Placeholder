"""Transformer image encoder."""

import torch

from src.models.base import Model, register_model


@register_model("transformer")
class Transformer(Model):
    """Encode an image represented as row, column, or patch tokens."""

    def __init__(
        self,
        token_dim: int = 28,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pass

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # ty: ignore[empty-body]
        """Return encoded image-token features."""
        pass
