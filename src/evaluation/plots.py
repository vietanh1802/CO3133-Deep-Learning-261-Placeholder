"""Required experiment-plot interfaces."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _prepare_output_path(output_path: str | Path) -> Path:
    """Create the parent directory for a figure and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_learning_curves(
    train_loss: Sequence[float],
    validation_loss: Sequence[float],
    output_path: str | Path,
) -> None:
    """Save training and validation curves."""
    if len(train_loss) == 0:
        raise ValueError("train_loss must not be empty")
    if len(train_loss) != len(validation_loss):
        raise ValueError("train_loss and validation_loss must have the same length")

    path = _prepare_output_path(output_path)
    epochs = np.arange(1, len(train_loss) + 1)

    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.plot(epochs, train_loss, marker="o", markersize=4, label="Training loss")
    axes.plot(epochs, validation_loss, marker="s", markersize=4, label="Validation loss")

    best_epoch = int(np.argmin(validation_loss))
    axes.axvline(
        best_epoch + 1,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"Best validation epoch ({best_epoch + 1})",
    )

    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.set_title("Training and validation loss")
    axes.legend()
    axes.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


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

    path = _prepare_output_path(output_path)

    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros(counts.shape, dtype=float),
        where=row_totals > 0,
    )

    size = max(6.0, 0.7 * len(class_names))
    figure, axes = plt.subplots(figsize=(size, size))
    image = axes.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axes, fraction=0.046, label="Proportion of true class")

    axes.set_xticks(np.arange(len(class_names)))
    axes.set_yticks(np.arange(len(class_names)))
    axes.set_xticklabels(class_names, rotation=45, ha="right")
    axes.set_yticklabels(class_names)
    axes.set_xlabel("Predicted label")
    axes.set_ylabel("True label")
    axes.set_title("Confusion matrix (row-normalized)")

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
                color="white" if normalized[row, column] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_prediction_examples(
    images: Sequence[np.ndarray],
    targets: Sequence[int],
    predictions: Sequence[int],
    confidences: Sequence[float],
    class_names: Sequence[str],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Save labeled examples selected by classification error analysis."""
    count = len(images)
    if count == 0:
        return
    if not (len(targets) == len(predictions) == len(confidences) == count):
        raise ValueError("images and prediction metadata must have the same length")

    path = _prepare_output_path(output_path)
    figure, axes = plt.subplots(1, count, figsize=(3.0 * count, 3.2), squeeze=False)
    for axis, image, target, prediction, confidence in zip(
        axes[0], images, targets, predictions, confidences, strict=True
    ):
        pixels = np.asarray(image)
        if pixels.ndim != 3 or pixels.shape[0] not in (1, 3):
            raise ValueError("each image must have shape [1, height, width] or [3, height, width]")
        pixels = np.moveaxis(pixels, 0, -1)
        minimum = float(pixels.min())
        maximum = float(pixels.max())
        pixels = (pixels - minimum) / (maximum - minimum) if maximum > minimum else pixels
        if pixels.shape[-1] == 1:
            axis.imshow(pixels[..., 0], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axis.imshow(pixels)
        axis.set_title(
            f"true: {class_names[target]}\npred: {class_names[prediction]} ({confidence:.2f})",
            fontsize=9,
        )
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
