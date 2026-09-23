"""Transformer image encoder."""

import math

import torch
from torch import nn

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
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        self.token_dim = token_dim
        self.output_dim = embedding_dim

        self.token_projection = nn.Linear(token_dim, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return encoded image-token features."""
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        batch_size, channels, height, width = images.shape
        tokens = images.permute(0, 2, 1, 3).reshape(batch_size, height, channels * width)
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"channels * image width must equal token_dim={self.token_dim}, "
                f"got {tokens.shape[-1]}"
            )

        embedded = self.token_projection(tokens)  # [batch, seq_len, embedding_dim]
        embedded = embedded + self._positional_encoding(embedded)
        embedded = self.embedding_dropout(embedded)

        encoded = self.encoder(embedded)  # [batch, seq_len, embedding_dim]
        return encoded.mean(dim=1)  # [batch, embedding_dim]

    @staticmethod
    def _positional_encoding(embedded: torch.Tensor) -> torch.Tensor:
        _, seq_len, dim = embedded.shape
        position = torch.arange(seq_len, device=embedded.device).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, dim, 2, device=embedded.device) * (-math.log(10000.0) / dim)
        )
        angle = position * frequency

        encoding = torch.zeros(seq_len, dim, device=embedded.device)
        encoding[:, 0::2] = torch.sin(angle)
        encoding[:, 1::2] = torch.cos(angle[:, : encoding[:, 1::2].shape[-1]])
        return encoding.unsqueeze(0)
