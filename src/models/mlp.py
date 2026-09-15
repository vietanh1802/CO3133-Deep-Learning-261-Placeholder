"""Multilayer perceptron image classifier."""

from collections.abc import Sequence

import torch

from src.models.base import ImageClassifier


class MLPClassifier(ImageClassifier):
    """Classify flattened images with one or more hidden layers."""

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dims: Sequence[int] = (256,),
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits."""
        raise NotImplementedError
