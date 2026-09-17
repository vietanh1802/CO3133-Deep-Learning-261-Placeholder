"""Linear-model image encoder."""

import torch

from src.models.base import Model, register_model


@register_model("linear")
class Linear(Model):
    """Prepare flattened image features for a linear classification head."""

    def __init__(self, input_dim: int = 28 * 28) -> None:
        super().__init__()
        pass

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # ty: ignore[empty-body]
        """Return flattened image features."""
        pass
