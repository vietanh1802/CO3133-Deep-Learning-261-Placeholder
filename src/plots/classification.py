"""Classification training curves and prediction examples."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.plots.common import prepare_output_path, save_figure  # noqa: E402
from src.plots.format import apply  # noqa: E402


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

    path = prepare_output_path(output_path)
    epochs = np.arange(1, len(train_loss) + 1)

    apply()
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

    figure.tight_layout()
    save_figure(figure, path)
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

    path = prepare_output_path(output_path)
    apply()
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
    save_figure(figure, path)
    plt.close(figure)


def plot_comparative_learning_curves(histories, output_path, *, loss_max, loss_ceiling=None):
    """Compare architectures with shared train/validation axes and the preset palette."""
    from src.plots.format import COLORS, GREEN

    apply()
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
    colors = (*COLORS, GREEN)
    upper_loss = loss_max * 1.03 if loss_ceiling is None else loss_ceiling
    if any(
        value > upper_loss
        for summary in histories.values()
        for key in ("train_loss", "validation_loss")
        for value in summary[key]
    ):
        raise ValueError("loss ceiling would hide recorded loss values")
    for index, (model, summary) in enumerate(histories.items()):
        color = colors[index % len(colors)]
        for axis, key in zip(axes, ("train_loss", "validation_loss"), strict=True):
            axis.plot(range(1, len(summary[key]) + 1), summary[key], color=color, label=model)
            axis.set(xlim=(1, summary["epochs"]), ylim=(0, upper_loss), xlabel="Epoch")
            if loss_ceiling is not None:
                axis.set_yticks(np.linspace(0, loss_ceiling, 4))
        axes[1].plot(
            summary["best_epoch"],
            summary["best_validation_loss"],
            "o",
            color=color,
            markersize=3,
        )
    axes[0].set(ylabel="Cross-entropy loss", title="Training")
    axes[1].set_title("Validation")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(histories))
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(figure, output_path)
    plt.close(figure)


def plot_shared_predictions(images, samples, class_names, output_path):
    """Show every model's predictions on the same deterministically selected images."""
    if not samples:
        return
    apply()
    rows = (len(samples) + 1) // 2
    figure = plt.figure(figsize=(10, 1.7 * rows), layout="constrained")
    grid = figure.add_gridspec(rows, 2)
    for index, (image, sample) in enumerate(zip(images, samples, strict=True)):
        cell = grid[index // 2, index % 2].subgridspec(1, 2, width_ratios=[1, 2.5], wspace=0.05)
        image_axis = figure.add_subplot(cell[0])
        text_axis = figure.add_subplot(cell[1])
        pixels = np.asarray(image)
        image_axis.imshow(pixels, cmap="gray" if pixels.ndim == 2 else None)
        image_axis.set_title(
            f"#{sample['sample_index']}: {class_names[sample['target']]}",
            fontsize=9,
        )
        image_axis.axis("off")
        lines = [sample["category"]]
        for model, prediction in sample["predictions"].items():
            correct = prediction == sample["target"]
            lines.append(
                f"{model}: {class_names[prediction]} ({'correct' if correct else 'wrong'})"
            )
        text_axis.text(0, 0.5, "\n".join(lines), va="center", fontsize=9)
        text_axis.axis("off")
    save_figure(figure, output_path)
    plt.close(figure)
