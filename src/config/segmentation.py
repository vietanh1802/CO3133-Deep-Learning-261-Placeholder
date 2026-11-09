"""Semantic-segmentation configuration loading and validation."""

from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from pathlib import Path
from typing import Any

from src.config.io import load_config_values, serialize_paths


@dataclass(frozen=True)
class SegmentationConfig:
    """Settings for one semantic-segmentation training run."""

    name: str
    model: str
    model_config: dict[str, Any] = field(default_factory=dict)
    dataset: str = "suim"
    seed: int = 42
    batch_size: int = 4
    accumulation_steps: int = 4
    learning_rate: float = 1e-3
    backbone_learning_rate: float | None = None
    weight_decay: float = 1e-4
    epochs: int = 40
    warmup_epochs: int = 5
    image_height: int = 240
    image_width: int = 320
    num_workers: int = 0
    augment: bool = True
    loss: str = "cross_entropy"
    evaluation_height: int | None = None
    evaluation_width: int | None = None
    data_dir: Path = Path("data/suim")
    checkpoint_dir: Path = Path("checkpoints")
    results_dir: Path = Path("results")

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        for key in ("batch_size", "accumulation_steps", "epochs", "image_height", "image_width"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.backbone_learning_rate is not None and (
            not isfinite(self.backbone_learning_rate) or self.backbone_learning_rate <= 0
        ):
            raise ValueError("backbone_learning_rate must be finite and positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must be in the range [0, epochs)")
        if not isinstance(self.augment, bool):
            raise ValueError("augment must be a boolean")
        if self.loss not in ("cross_entropy", "ce_dice"):
            raise ValueError("loss must be cross_entropy or ce_dice")
        if (self.evaluation_height is None) != (self.evaluation_width is None):
            raise ValueError("evaluation_height and evaluation_width must be set together")
        for size in (self.evaluation_height, self.evaluation_width):
            if size is not None and size <= 0:
                raise ValueError("evaluation dimensions must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready representation."""
        return serialize_paths(asdict(self))


def load_segmentation_config(path: str | Path) -> SegmentationConfig:
    """Load a segmentation configuration file, rejecting unknown or missing keys."""
    return SegmentationConfig(**load_config_values(path, fields(SegmentationConfig)))
