"""Experiment configuration contract."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings that must be captured with every experiment result."""

    name: str
    model: str
    dataset: str = "fashion_mnist"
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 10
    num_workers: int = 0
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    results_dir: Path = Path("results")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready representation."""
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration file."""
    raise NotImplementedError
