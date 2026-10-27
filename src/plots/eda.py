"""Plot computed EDA summaries as matching PNG and PDF figures."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.eda.classification import ClassBalance, DatasetProfile, IntensitySummary  # noqa: E402
from src.plots.common import prepare_output_path, save_figure  # noqa: E402
from src.plots.format import BLUE, BLUE_RED, CLASS_COLORS, RED, apply  # noqa: E402


def _as_displayable(image: np.ndarray) -> np.ndarray:
    """Rescale one [H, W, C] image to [0, 1] so float prototypes render correctly."""
    pixels = np.asarray(image, dtype=np.float64)
    minimum = float(np.min(pixels))
    maximum = float(np.max(pixels))
    if maximum <= minimum:
        return np.zeros_like(pixels)
    return (pixels - minimum) / (maximum - minimum)


def _show_image(axes: plt.Axes, image: np.ndarray) -> None:
    """Draw a [H, W, C] image on an axis, handling grayscale and RGB alike."""
    pixels = _as_displayable(image)
    if pixels.shape[-1] == 1:
        axes.imshow(pixels[..., 0], cmap="gray", vmin=0.0, vmax=1.0)
    else:
        axes.imshow(pixels)
    axes.axis("off")


def plot_class_distribution(
    balance: ClassBalance,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Save the required class-distribution bar chart with counts annotated."""
    path = prepare_output_path(output_path)
    names = balance.class_names
    counts = np.asarray(balance.counts)

    apply()
    figure, axes = plt.subplots(figsize=(max(7.0, 0.9 * len(names)), 4.5))
    bars = axes.bar(
        names, counts, color=[CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(names))]
    )
    axes.bar_label(bars, fmt="%d", fontsize=10, padding=2)

    axes.set_xlabel("Class")
    axes.set_ylabel("Number of samples")
    axes.set_title(title)
    axes.set_ylim(0, counts.max() * 1.12)
    axes.set_xticks(range(len(names)), names, rotation=30, ha="right")

    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def plot_split_class_distribution(
    train_balance: ClassBalance,
    test_balance: ClassBalance,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Compare train and test class proportions across unequal split sizes."""
    if train_balance.class_names != test_balance.class_names:
        raise ValueError("both splits must use the same class names")

    path = prepare_output_path(output_path)
    names = train_balance.class_names
    positions = np.arange(len(names))
    width = 0.4

    apply()
    figure, axes = plt.subplots(figsize=(max(7.0, 0.9 * len(names)), 4.5))
    axes.bar(
        positions - width / 2,
        train_balance.proportions,
        width,
        label="Train",
        color=BLUE,
    )
    axes.bar(
        positions + width / 2,
        test_balance.proportions,
        width,
        label="Test",
        color=RED,
    )

    axes.set_xticks(positions)
    axes.set_xticklabels(names, rotation=30, ha="right")
    axes.set_xlabel("Class")
    axes.set_ylabel("Proportion of split")
    axes.set_title(title, pad=35)
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)

    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def plot_intensity_histogram(
    intensity: IntensitySummary,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Save the pixel-intensity histogram that motivates the normalization constants."""
    path = prepare_output_path(output_path)
    edges = np.asarray(intensity.bin_edges)
    counts = np.asarray(intensity.histogram)
    centers = (edges[:-1] + edges[1:]) / 2.0

    apply()
    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    axes.bar(centers, counts, width=np.diff(edges), color=BLUE_RED(centers / 255.0), align="center")

    # Log scale keeps the background spike from hiding foreground counts.
    axes.set_yscale("log")
    axes.axvline(
        intensity.mean,
        color=RED,
        linestyle="--",
        linewidth=1,
        label=f"Mean {intensity.mean:.1f}",
    )

    axes.set_xlabel("Pixel intensity (0-255)")
    axes.set_ylabel("Pixel count (log scale)")
    axes.set_title(title, pad=35)
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)

    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def plot_class_mean_images(
    profile: DatasetProfile,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Save the average image of each class, side by side."""
    path = prepare_output_path(output_path)
    names = profile.balance.class_names
    columns = min(5, len(names))
    rows = int(np.ceil(len(names) / columns))

    apply()
    figure, axes = plt.subplots(
        rows,
        columns,
        # Reserve height for the title and row labels.
        figsize=(2.0 * columns, 2.4 * rows + 0.5),
        squeeze=False,
    )
    for index, axis in enumerate(axes.flat):
        if index >= len(names):
            axis.axis("off")
            continue
        _show_image(axis, profile.class_mean_images[index])
        axis.set_title(names[index], fontsize=8)

    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_figure(figure, path)
    plt.close(figure)


def plot_class_similarity(
    profile: DatasetProfile,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Save the class-prototype similarity heatmap, annotated with its values."""
    path = prepare_output_path(output_path)
    names = profile.balance.class_names
    similarity = np.asarray(profile.class_similarity)

    size = max(6.0, 0.75 * len(names))
    apply()
    figure, axes = plt.subplots(figsize=(size, size))
    image = axes.imshow(similarity, cmap=BLUE_RED, vmin=-1.0, vmax=1.0)
    figure.colorbar(image, ax=axes, fraction=0.046, label="Cosine similarity")

    axes.set_xticks(np.arange(len(names)))
    axes.set_yticks(np.arange(len(names)))
    axes.set_xticklabels(names, rotation=45, ha="right")
    axes.set_yticklabels(names)
    axes.set_title(title)

    for row in range(len(names)):
        for column in range(len(names)):
            value = similarity[row, column]
            axes.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(value) > 0.85 else "black",
            )

    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def plot_foreground_coverage(
    profile: DatasetProfile,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Save per-class foreground coverage: how much of the frame the object fills."""
    path = prepare_output_path(output_path)
    names = profile.balance.class_names
    coverage = np.asarray(profile.intensity.per_class_foreground_fraction)

    order = np.argsort(coverage)[::-1]
    sorted_names = [names[index] for index in order]

    apply()
    figure, axes = plt.subplots(figsize=(max(7.0, 0.9 * len(names)), 4.5))
    bars = axes.barh(
        sorted_names, coverage[order], color=[CLASS_COLORS[i % len(CLASS_COLORS)] for i in order]
    )
    axes.bar_label(bars, fmt="%.2f", fontsize=10, padding=2)

    axes.invert_yaxis()
    axes.set_xlabel("Foreground fraction")
    axes.set_ylabel("Class")
    axes.set_title(title)
    axes.set_xlim(0, min(1.0, float(coverage.max()) * 1.18))

    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def plot_representative_samples(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
    *,
    title: str,
    samples_per_class: int = 3,
) -> None:
    """Save the first examples per class in a compact grid of class panels."""
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

    path = prepare_output_path(output_path)
    batch = np.asarray(images)
    if batch.ndim == 3:
        batch = batch[..., np.newaxis]
    values = np.asarray(labels).astype(np.int64).ravel()

    apply()
    columns = min(5, len(class_names))
    rows = int(np.ceil(len(class_names) / columns))
    figure = plt.figure(figsize=(2.0 * columns, 1.5 * rows))
    grid = figure.add_gridspec(rows, columns, hspace=0.45, wspace=0.15)
    for class_id, name in enumerate(class_names):
        cell = grid[class_id // columns, class_id % columns].subgridspec(
            1, samples_per_class, wspace=0.04
        )
        indices = np.flatnonzero(values == class_id)[:samples_per_class]
        for column in range(samples_per_class):
            axis = figure.add_subplot(cell[0, column])
            if column >= len(indices):
                axis.axis("off")
                continue
            _show_image(axis, batch[indices[column]])
            if column == samples_per_class // 2:
                axis.set_title(name)

    figure.suptitle(title)
    figure.subplots_adjust(top=0.78, bottom=0.06, left=0.02, right=0.98)
    save_figure(figure, path)
    plt.close(figure)
