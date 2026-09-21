"""Multilayer perceptron image encoder."""

from collections.abc import Sequence

import torch
from torch import nn

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
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_dims or any(hidden_dim <= 0 for hidden_dim in hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        layers: list[nn.Module] = [nn.Flatten()]
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        self.input_dim = input_dim
        self.output_dim = previous_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return encoded image features."""
        features = self.layers(images)
        if features.shape[-1] != self.output_dim:
            raise RuntimeError("MLP produced an unexpected feature dimension")
        return features
