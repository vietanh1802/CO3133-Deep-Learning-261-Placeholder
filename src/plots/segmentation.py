"""Segmentation training-curve figures."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.plots.common import prepare_output_path, save_figure  # noqa: E402
from src.plots.format import COLORS, apply  # noqa: E402


def plot_segmentation_curves(
    train_loss: Sequence[float],
    validation_loss: Sequence[float],
    validation_miou: Sequence[float],
    output_path: str | Path,
) -> None:
    """Save loss and validation-mIoU curves, marking the highest-mIoU epoch."""
    if len(train_loss) == 0:
        raise ValueError("train_loss must not be empty")
    if not len(train_loss) == len(validation_loss) == len(validation_miou):
        raise ValueError("train_loss, validation_loss and validation_miou must match in length")

    path = prepare_output_path(output_path)
    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch = int(np.argmax(validation_miou)) + 1

    apply()
    figure, (loss_axes, miou_axes) = plt.subplots(1, 2)
    loss_axes.plot(epochs, train_loss, marker="o", label="Training loss")
    loss_axes.plot(epochs, validation_loss, marker="s", label="Validation loss")
    loss_axes.set_ylabel("Cross-entropy loss")
    loss_axes.set_title("Training and validation loss")
    miou_axes.plot(epochs, validation_miou, marker="s", color=COLORS[0])
    miou_axes.set_ylabel("Validation mIoU")
    miou_axes.set_title("Validation mIoU")

    for axes in (loss_axes, miou_axes):
        axes.axvline(
            best_epoch,
            color=COLORS[2],
            linestyle="--",
            linewidth=1,
            label=f"Selected epoch ({best_epoch}, highest mIoU)",
        )
        axes.set_xlabel("Epoch")
        axes.legend()

    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)
