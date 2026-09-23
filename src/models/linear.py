"""Linear-model image encoder."""

import torch
from torch import nn

from src.models.base import Model, register_model


@register_model("linear")
class Linear(Model):
    """Prepare flattened image features for a linear classification head."""

    def __init__(self, input_dim: int = 28 * 28) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.flatten = nn.Flatten()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return flattened image features."""
        features = self.flatten(images)
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} features per sample, got {features.shape[-1]}"
            )
        return features
