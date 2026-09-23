"""Vanilla RNN image-sequence encoder."""

import torch
from torch import nn

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
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return encoded sequence features."""
        if images.dim() == 4:
            if images.shape[1] != 1:
                raise ValueError(f"expected a single-channel image, got {images.shape[1]} channels")
            sequence = images.squeeze(1)
        elif images.dim() == 3:
            sequence = images
        else:
            raise ValueError(f"expected a 3D or 4D image batch, got {images.dim()} dimensions")
        if sequence.shape[-1] != self.input_size:
            raise ValueError(
                f"expected {self.input_size} features per timestep, got {sequence.shape[-1]}"
            )
        outputs, _ = self.rnn(sequence)
        return outputs[:, -1, :]
