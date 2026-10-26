"""Experiment configuration loading and validation."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from src.config.io import load_config_values, serialize_paths


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
        return serialize_paths(asdict(self))


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration file."""
    return ExperimentConfig(**load_config_values(path, fields(ExperimentConfig)))
