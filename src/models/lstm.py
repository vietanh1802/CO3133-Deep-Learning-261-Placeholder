"""LSTM image-sequence encoder."""

import torch
from torch import nn

from src.models.base import Model, register_model


class LSTMCell(nn.Module):
    """Compute one custom LSTM recurrent step."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.project_input = nn.Linear(input_size, 4 * hidden_size)
        self.project_hidden = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next hidden and cell states for one timestep."""
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_size:
            raise ValueError(f"inputs must have shape [batch, {self.input_size}]")
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden must have shape [batch, {self.hidden_size}]")
        if cell.shape != hidden.shape:
            raise ValueError("hidden and cell must have the same shape")
        if inputs.shape[0] != hidden.shape[0]:
            raise ValueError("inputs and hidden must have the same batch size")

        gates = self.project_input(inputs) + self.project_hidden(hidden)
        raw_input, raw_forget, raw_candidate, raw_output = gates.chunk(4, dim=-1)

        input_gate = torch.sigmoid(raw_input)
        forget_gate = torch.sigmoid(raw_forget)
        candidate = torch.tanh(raw_candidate)
        output_gate = torch.sigmoid(raw_output)

        next_cell = forget_gate * cell + input_gate * candidate
        return output_gate * torch.tanh(next_cell), next_cell


@register_model("lstm")
class LSTM(Model):
    """Encode image sequences with a custom stacked LSTM."""

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
                LSTMCell(input_size, hidden_size),
                *[LSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)],
            ]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return the final hidden features of the last recurrent layer."""
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        if images.shape[2] == 0:
            raise ValueError("sequence length must be positive")

        batch_size, channels, height, width = images.shape
        layer_input = images.permute(0, 2, 1, 3).reshape(batch_size, height, channels * width)
        if layer_input.shape[-1] != self.input_size:
            raise ValueError(
                f"channels * image width must equal input_size={self.input_size}, "
                f"got {layer_input.shape[-1]}"
            )
        hidden = layer_input.new_zeros(layer_input.shape[0], self.hidden_size)

        for layer_index, cell_module in enumerate(self.cells):
            hidden = layer_input.new_zeros(layer_input.shape[0], self.hidden_size)
            cell_state = layer_input.new_zeros(layer_input.shape[0], self.hidden_size)
            outputs: list[torch.Tensor] = []

            for timestep in range(layer_input.shape[1]):
                hidden, cell_state = cell_module(layer_input[:, timestep, :], hidden, cell_state)
                outputs.append(hidden)

            layer_output = torch.stack(outputs, dim=1)
            if layer_index < self.num_layers - 1:
                layer_output = self.inter_layer_dropout(layer_output)
            layer_input = layer_output

        return hidden
