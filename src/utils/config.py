"""Experiment configuration loading and validation."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings that must be captured with every experiment result."""

    name: str
    model: str
    model_config: dict[str, Any] = field(default_factory=dict)
    dataset: str = "fashion_mnist"
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 10
    num_workers: int = 0
    val_fraction: float = 0.1
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    results_dir: Path = Path("results")

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.dataset.strip():
            raise ValueError("dataset must not be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in the range (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready representation."""
        values = asdict(self)
        for key in ("data_dir", "checkpoint_dir", "results_dir"):
            values[key] = str(values[key])
        return values


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration file."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)

    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    valid_keys = {field.name for field in fields(ExperimentConfig)}
    unknown_keys = set(raw) - valid_keys
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"unknown configuration keys: {unknown}")

    missing_keys = {"name", "model"} - set(raw)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"missing required configuration keys: {missing}")

    for key in ("data_dir", "checkpoint_dir", "results_dir"):
        if key in raw:
            raw[key] = Path(raw[key])

    return ExperimentConfig(**raw)
