"""Required experiment-plot interfaces."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def plot_learning_curves(
    train_loss: Sequence[float],
    validation_loss: Sequence[float],
    output_path: str | Path,
) -> None:
    """Save training and validation curves."""
    raise NotImplementedError


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
) -> None:
    """Save a labeled confusion-matrix figure."""
    raise NotImplementedError
