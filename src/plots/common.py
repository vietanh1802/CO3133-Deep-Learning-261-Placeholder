"""Shared figure output paths and confusion-matrix rendering."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.plots.format import BLUE_RED, apply  # noqa: E402


def prepare_output_path(output_path: str | Path) -> Path:
    """Create the parent directory for a figure and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(figure: Figure, output_path: str | Path) -> None:
    """Save a PDF and 400-dpi PNG, retaining any explicitly requested extra format."""
    path = prepare_output_path(output_path)
    outputs = [path.with_suffix(".png"), path.with_suffix(".pdf")]
    if path.suffix and path not in outputs:
        outputs.insert(0, path)
    for output in outputs:
        figure.savefig(output, dpi=400, bbox_inches="tight", facecolor="white")


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
) -> None:
    """Save a labeled confusion-matrix figure."""
    counts = np.asarray(matrix)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("matrix must be square")
    if len(class_names) != counts.shape[0]:
        raise ValueError("class_names must have one entry per class")

    path = prepare_output_path(output_path)

    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros(counts.shape, dtype=float),
        where=row_totals > 0,
    )

    size = max(6.0, 0.7 * len(class_names))
    apply()
    figure, axes = plt.subplots(figsize=(size, size))
    image = axes.imshow(normalized, cmap=BLUE_RED, vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axes, fraction=0.046, label="Proportion of true class")

    axes.set_xticks(np.arange(len(class_names)))
    axes.set_yticks(np.arange(len(class_names)))
    axes.set_xticklabels(class_names, rotation=45, ha="right")
    axes.set_yticklabels(class_names)
    axes.set_xlabel("Predicted label")
    axes.set_ylabel("True label")
    axes.set_title("Confusion matrix")

    threshold = 0.5
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            axes.text(
                column,
                row,
                f"{counts[row, column]:d}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(normalized[row, column] - 0.5) > threshold * 0.8 else "black",
            )

    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)
