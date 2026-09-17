"""Multilayer perceptron image encoder."""

from collections.abc import Sequence

import torch

from src.models.base import Model, register_model


@register_model("mlp")
class MLP(Model):
    """Encode flattened images with one or more hidden layers."""

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dims: Sequence[int] = (256,),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pass

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # ty: ignore[empty-body]
        """Return encoded image features."""
        pass
