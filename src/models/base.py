"""Shared model contract and construction boundary."""

from collections.abc import Mapping
from typing import Any

from torch import nn


class ImageClassifier(nn.Module):
    """Base contract for A1 models that map image batches to class logits."""


def build_model(name: str, config: Mapping[str, Any]) -> nn.Module:
    """Construct one A1 model from its name and model-specific settings."""
    raise NotImplementedError
