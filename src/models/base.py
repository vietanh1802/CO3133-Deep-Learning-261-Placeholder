"""Shared model contract and construction boundary."""

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from torch import nn


class Model(nn.Module):
    """Base type for registered reusable model components."""

    output_dim: int


ModelT = TypeVar("ModelT", bound=type[Model])
_MODEL_REGISTRY: dict[str, type[Model]] = {}


def register_model(name: str) -> Callable[[ModelT], ModelT]:
    """Register an image-classifier class under a factory name."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Model name cannot be empty")

    def decorator(model_class: ModelT) -> ModelT:
        if key in _MODEL_REGISTRY:
            raise ValueError(f"Model {key!r} is already registered")
        _MODEL_REGISTRY[key] = model_class
        return model_class

    return decorator


def build_model(name: str, config: Mapping[str, Any]) -> Model:
    """Construct one registered model component."""
    key = name.strip().lower()
    if key not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Available models: {available}")
    return _MODEL_REGISTRY[key](**dict(config))
