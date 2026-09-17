"""Vanilla RNN image-sequence encoder."""

import torch

from src.models.base import Model, register_model


@register_model("rnn")
class RNN(Model):
    """Interpret image rows, columns, or patches as a sequence."""

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pass

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # ty: ignore[empty-body]
        """Return encoded sequence features."""
        pass
