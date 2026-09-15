"""GRU image-sequence classifier."""

import torch

from src.models.base import ImageClassifier


class GRUClassifier(ImageClassifier):
    """Optional GRU alternative using the same sequence representation as the LSTM."""

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits."""
        raise NotImplementedError
