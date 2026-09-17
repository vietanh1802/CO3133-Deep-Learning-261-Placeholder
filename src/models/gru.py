"""GRU image-sequence encoder."""

import torch
from torch import nn

from src.models.base import Model, register_model


class GRUCell(nn.Module):
    """Compute one custom GRU recurrent step."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.project_input = nn.Linear(input_size, 3 * hidden_size)
        self.project_hidden = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Return the next hidden state for one timestep."""
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_size:
            raise ValueError(f"inputs must have shape [batch, {self.input_size}]")
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden must have shape [batch, {self.hidden_size}]")
        if inputs.shape[0] != hidden.shape[0]:
            raise ValueError("inputs and hidden must have the same batch size")

        projected_input = self.project_input(inputs)
        projected_hidden = self.project_hidden(hidden)
        input_reset, input_update, input_candidate = projected_input.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_candidate = projected_hidden.chunk(3, dim=-1)

        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_candidate + reset_gate * hidden_candidate)

        return (1 - update_gate) * candidate + update_gate * hidden


@register_model("gru")
class GRU(Model):
    """Encode image sequences with a custom stacked GRU."""

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
            raise ValueError("dropout must be in the range [0, 1)")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = hidden_size
        self.num_layers = num_layers
        self.dropout_probability = dropout
        self.inter_layer_dropout = nn.Dropout(dropout)
        self.cells = nn.ModuleList(
            [
                GRUCell(input_size, hidden_size),
                *[GRUCell(hidden_size, hidden_size) for _ in range(num_layers - 1)],
            ]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return the final hidden features of the last recurrent layer."""
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("images must have shape [batch, 1, sequence_length, input_size]")
        if images.shape[-1] != self.input_size:
            raise ValueError(f"image width must equal input_size={self.input_size}")
        if images.shape[2] == 0:
            raise ValueError("sequence length must be positive")

        layer_input = images.squeeze(1)
        hidden = layer_input.new_zeros(layer_input.shape[0], self.hidden_size)

        for layer_index, cell in enumerate(self.cells):
            hidden = layer_input.new_zeros(layer_input.shape[0], self.hidden_size)
            outputs: list[torch.Tensor] = []

            for timestep in range(layer_input.shape[1]):
                hidden = cell(layer_input[:, timestep, :], hidden)
                outputs.append(hidden)

            layer_output = torch.stack(outputs, dim=1)
            if layer_index < self.num_layers - 1:
                layer_output = self.inter_layer_dropout(layer_output)
            layer_input = layer_output

        return hidden
