"""Semantic-segmentation configuration loading and validation."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PATH_KEYS = ("data_dir", "checkpoint_dir", "results_dir")


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
    weight_decay: float = 1e-4
    epochs: int = 40
    warmup_epochs: int = 5
    image_height: int = 240
    image_width: int = 320
    num_workers: int = 0
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
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must be in the range [0, epochs)")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready representation."""
        values = asdict(self)
        for key in PATH_KEYS:
            values[key] = str(values[key])
        return values


def load_segmentation_config(path: str | Path) -> SegmentationConfig:
    """Load a segmentation configuration file, rejecting unknown or missing keys."""
    with Path(path).open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)

    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    unknown_keys = set(raw) - {item.name for item in fields(SegmentationConfig)}
    if unknown_keys:
        raise ValueError(f"unknown configuration keys: {', '.join(sorted(unknown_keys))}")

    missing_keys = {"name", "model"} - set(raw)
    if missing_keys:
        raise ValueError(f"missing required configuration keys: {', '.join(sorted(missing_keys))}")

    for key in PATH_KEYS:
        if key in raw:
            raw[key] = Path(raw[key])

    return SegmentationConfig(**raw)
