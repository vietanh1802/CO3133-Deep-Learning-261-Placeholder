"""Figures for the Assignment 1 exploratory data analysis.

Every function here takes an already-computed summary from :mod:`src.data.eda` and
writes exactly one PNG, so a figure can never disagree with the JSON summary.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.eda import ClassBalance, DatasetProfile, IntensitySummary  # noqa: E402


def _prepare_output_path(output_path: str | Path) -> Path:
    """Create the parent directory for a figure and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
    path = _prepare_output_path(output_path)
    names = balance.class_names
    counts = np.asarray(balance.counts)

    figure, axes = plt.subplots(figsize=(max(7.0, 0.9 * len(names)), 4.5))
    bars = axes.bar(names, counts, color="steelblue")
    axes.bar_label(bars, fmt="%d", fontsize=8, padding=2)

    # A flat line at the perfectly balanced count makes any skew obvious at a glance.
    axes.axhline(
        counts.mean(),
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"Balanced count ({counts.mean():.0f})",
    )

    axes.set_xlabel("Class")
    axes.set_ylabel("Number of samples")
    axes.set_title(f"{title} (imbalance ratio {balance.imbalance_ratio:.2f})")
    axes.set_ylim(0, counts.max() * 1.12)
    axes.tick_params(axis="x", rotation=30)
    axes.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_split_class_distribution(
    train_balance: ClassBalance,
    test_balance: ClassBalance,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Compare train and test class proportions to check for distribution shift.

    Proportions rather than raw counts, because the two splits have different sizes.
    """
    if train_balance.class_names != test_balance.class_names:
        raise ValueError("both splits must use the same class names")

    path = _prepare_output_path(output_path)
    names = train_balance.class_names
    positions = np.arange(len(names))
    width = 0.4

    figure, axes = plt.subplots(figsize=(max(7.0, 0.9 * len(names)), 4.5))
    axes.bar(
        positions - width / 2,
        train_balance.proportions,
        width,
        label="Train",
        color="steelblue",
    )
    axes.bar(
        positions + width / 2,
        test_balance.proportions,
        width,
        label="Test",
        color="indianred",
    )

    axes.set_xticks(positions)
    axes.set_xticklabels(names, rotation=30, ha="right")
    axes.set_xlabel("Class")
    axes.set_ylabel("Proportion of split")
    axes.set_title(title)
    axes.legend()
    axes.grid(True, axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_intensity_histogram(
    intensity: IntensitySummary,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Save the pixel-intensity histogram that motivates the normalization constants."""
    path = _prepare_output_path(output_path)
    edges = np.asarray(intensity.bin_edges)
    counts = np.asarray(intensity.histogram)
    centers = (edges[:-1] + edges[1:]) / 2.0

    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    axes.bar(centers, counts, width=np.diff(edges), color="steelblue", align="center")

    # Log scale: the background spike at 0 is orders of magnitude taller than the rest,
    # and would otherwise flatten the entire foreground range into the axis.
    axes.set_yscale("log")
    axes.axvline(
        intensity.mean,
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean intensity ({intensity.mean:.1f})",
    )

    axes.set_xlabel("Pixel intensity (0-255)")
    axes.set_ylabel("Pixel count (log scale)")
    axes.set_title(f"{title} (std {intensity.std:.1f})")
    axes.legend()
    axes.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_class_mean_images(
    profile: DatasetProfile,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Save the average image of each class, side by side."""
    path = _prepare_output_path(output_path)
    names = profile.balance.class_names
    columns = min(5, len(names))
    rows = int(np.ceil(len(names) / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        # The extra half inch of height leaves room for the suptitle, so the first row
        # of images does not run into the second row's class labels.
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
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_class_similarity(
    profile: DatasetProfile,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Save the class-prototype similarity heatmap, annotated with its values."""
    path = _prepare_output_path(output_path)
    names = profile.balance.class_names
    similarity = np.asarray(profile.class_similarity)

    size = max(6.0, 0.75 * len(names))
    figure, axes = plt.subplots(figsize=(size, size))
    image = axes.imshow(similarity, cmap="coolwarm", vmin=-1.0, vmax=1.0)
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
                fontsize=7,
                color="white" if abs(value) > 0.6 else "black",
            )

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_foreground_coverage(
    profile: DatasetProfile,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Save per-class foreground coverage: how much of the frame the object fills."""
    path = _prepare_output_path(output_path)
    names = profile.balance.class_names
    coverage = np.asarray(profile.intensity.per_class_foreground_fraction)

    order = np.argsort(coverage)[::-1]
    sorted_names = [names[index] for index in order]

    figure, axes = plt.subplots(figsize=(max(7.0, 0.9 * len(names)), 4.5))
    bars = axes.barh(sorted_names, coverage[order], color="seagreen")
    axes.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    axes.invert_yaxis()
    axes.set_xlabel("Fraction of pixels above the foreground threshold")
    axes.set_ylabel("Class")
    axes.set_title(title)
    axes.set_xlim(0, min(1.0, float(coverage.max()) * 1.18))
    axes.grid(True, axis="x", alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
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
    """Save a grid of real examples, one row per class.

    Several examples per class rather than one, so the reader can see within-class
    variation instead of judging a class by a single lucky image.
    """
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

    path = _prepare_output_path(output_path)
    batch = np.asarray(images)
    if batch.ndim == 3:
        batch = batch[..., np.newaxis]
    values = np.asarray(labels).astype(np.int64).ravel()

    figure, axes = plt.subplots(
        len(class_names),
        samples_per_class,
        figsize=(1.5 * samples_per_class, 1.5 * len(class_names)),
        squeeze=False,
    )
    for class_id, name in enumerate(class_names):
        indices = np.flatnonzero(values == class_id)[:samples_per_class]
        for column, axis in enumerate(axes[class_id]):
            if column >= len(indices):
                axis.axis("off")
                continue
            _show_image(axis, batch[indices[column]])
            if column == 0:
                # One label per row, written as a left-hand y-label so it does not
                # collide with the images themselves.
                axis.axis("on")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
