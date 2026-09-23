"""Exploratory data analysis for image-classification datasets.

This module only *computes* statistics; rendering lives in :mod:`src.data.eda_plots`
and orchestration lives in ``scripts/eda/``. Keeping the three apart means every number
reported in the assignment can be unit-tested without touching matplotlib.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MNIST and Fashion-MNIST both draw a light object on a black background, so a single
# uint8 cut-off separates "ink" from background well enough to compare classes.
FOREGROUND_THRESHOLD = 32

# 32 bins over the uint8 range keeps the histogram readable while still showing the
# background spike at 0 that motivates the mean=0.5 / std=0.5 normalization.
INTENSITY_BINS = 32


@dataclass(frozen=True)
class ImageShapeSummary:
    """Input geometry and raw value range of one split."""

    num_samples: int
    height: int
    width: int
    channels: int
    dtype: str
    min_value: float
    max_value: float


@dataclass(frozen=True)
class ClassBalance:
    """Per-class sample counts and the resulting imbalance."""

    class_names: tuple[str, ...]
    counts: tuple[int, ...]
    proportions: tuple[float, ...]
    imbalance_ratio: float
    rarest_class: str
    most_common_class: str


@dataclass(frozen=True)
class IntensitySummary:
    """Pixel-intensity statistics, globally and per class."""

    mean: float
    std: float
    histogram: tuple[int, ...]
    bin_edges: tuple[float, ...]
    per_class_mean_intensity: tuple[float, ...]
    per_class_foreground_fraction: tuple[float, ...]


@dataclass(frozen=True)
class DatasetProfile:
    """Every EDA figure and number the Assignment 1 report needs for one split."""

    dataset: str
    split: str
    shape: ImageShapeSummary
    balance: ClassBalance
    intensity: IntensitySummary
    class_mean_images: np.ndarray  # [num_classes, height, width, channels], float64
    class_similarity: np.ndarray  # [num_classes, num_classes], float64 in [-1, 1]


def as_image_batch(images: np.ndarray) -> np.ndarray:
    """Return images as [num_samples, height, width, channels].

    Grayscale datasets expose ``[N, H, W]`` while CIFAR-10 exposes ``[N, H, W, 3]``;
    every statistic below is written against the four-dimensional form only.
    """
    batch = np.asarray(images)
    if batch.ndim == 3:
        batch = batch[..., np.newaxis]
    if batch.ndim != 4:
        raise ValueError("images must have shape [N, H, W] or [N, H, W, C]")
    if batch.shape[0] == 0:
        raise ValueError("images must not be empty")
    return batch


def summarize_image_shape(images: np.ndarray) -> ImageShapeSummary:
    """Describe input size and value range, as required by the EDA section."""
    batch = as_image_batch(images)
    num_samples, height, width, channels = batch.shape
    return ImageShapeSummary(
        num_samples=int(num_samples),
        height=int(height),
        width=int(width),
        channels=int(channels),
        dtype=str(batch.dtype),
        min_value=float(np.min(batch)),
        max_value=float(np.max(batch)),
    )


def summarize_class_balance(labels: np.ndarray, class_names: Sequence[str]) -> ClassBalance:
    """Count samples per class and quantify imbalance as max count / min count."""
    values = np.asarray(labels).astype(np.int64).ravel()
    if values.size == 0:
        raise ValueError("labels must not be empty")
    if len(class_names) == 0:
        raise ValueError("class_names must not be empty")
    if int(np.min(values)) < 0 or int(np.max(values)) >= len(class_names):
        raise ValueError(f"labels must contain values in [0, {len(class_names)})")

    counts = np.bincount(values, minlength=len(class_names))
    proportions = counts / values.size

    # A ratio of 1.00 means perfectly balanced. The assignment asks us to state this
    # explicitly rather than leave the reader to infer it from the bar chart.
    smallest = int(np.min(counts))
    imbalance_ratio = int(np.max(counts)) / smallest if smallest > 0 else float("inf")

    return ClassBalance(
        class_names=tuple(class_names),
        counts=tuple(int(count) for count in counts),
        proportions=tuple(float(value) for value in proportions),
        imbalance_ratio=float(imbalance_ratio),
        rarest_class=class_names[int(np.argmin(counts))],
        most_common_class=class_names[int(np.argmax(counts))],
    )


def summarize_intensity(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
) -> IntensitySummary:
    """Summarize pixel intensities globally and per class.

    The global histogram justifies the normalization constants used in
    :mod:`src.data.transform`; the per-class numbers show how much of each image the
    object actually occupies, which differs sharply between e.g. Bag and Sandal.
    """
    batch = as_image_batch(images)
    values = np.asarray(labels).astype(np.int64).ravel()
    if values.size != batch.shape[0]:
        raise ValueError("images and labels must have the same number of samples")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    pixels = batch.reshape(batch.shape[0], -1)
    histogram, bin_edges = np.histogram(batch, bins=INTENSITY_BINS, range=(0.0, 255.0))

    mean_per_sample = pixels.mean(axis=1)
    foreground_per_sample = (pixels > FOREGROUND_THRESHOLD).mean(axis=1)

    per_class_mean = np.zeros(num_classes, dtype=np.float64)
    per_class_foreground = np.zeros(num_classes, dtype=np.float64)
    for class_id in range(num_classes):
        selected = values == class_id
        if not np.any(selected):
            continue
        per_class_mean[class_id] = float(mean_per_sample[selected].mean())
        per_class_foreground[class_id] = float(foreground_per_sample[selected].mean())

    return IntensitySummary(
        mean=float(np.mean(batch)),
        std=float(np.std(batch)),
        histogram=tuple(int(count) for count in histogram),
        bin_edges=tuple(float(edge) for edge in bin_edges),
        per_class_mean_intensity=tuple(float(value) for value in per_class_mean),
        per_class_foreground_fraction=tuple(float(value) for value in per_class_foreground),
    )


def compute_class_mean_images(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    """Average every image of a class into one [H, W, C] prototype per class."""
    batch = as_image_batch(images).astype(np.float64)
    values = np.asarray(labels).astype(np.int64).ravel()
    if values.size != batch.shape[0]:
        raise ValueError("images and labels must have the same number of samples")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    means = np.zeros((num_classes, *batch.shape[1:]), dtype=np.float64)
    for class_id in range(num_classes):
        selected = values == class_id
        if np.any(selected):
            means[class_id] = batch[selected].mean(axis=0)
    return means


def compute_class_similarity(class_mean_images: np.ndarray) -> np.ndarray:
    """Return a [num_classes, num_classes] cosine similarity between class prototypes.

    The average of all prototypes is subtracted first. Without that step every
    prototype shares the same bright centre and all similarities collapse to roughly
    0.9; after centring, the matrix measures what actually *distinguishes* one class
    from another, which is what predicts the confusion pairs seen at test time.
    """
    means = np.asarray(class_mean_images, dtype=np.float64)
    if means.ndim < 2:
        raise ValueError("class_mean_images must have shape [num_classes, ...]")

    centered = means - means.mean(axis=0, keepdims=True)
    flat = centered.reshape(means.shape[0], -1)

    norms = np.linalg.norm(flat, axis=1)
    # Guard classes with no samples (an all-zero prototype) against division by zero.
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    normalized = flat / safe_norms[:, np.newaxis]
    return normalized @ normalized.T


def profile_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    dataset: str,
    split: str,
) -> DatasetProfile:
    """Run the full EDA pass over one split and collect it into a single object."""
    class_mean_images = compute_class_mean_images(images, labels, num_classes=len(class_names))
    return DatasetProfile(
        dataset=dataset,
        split=split,
        shape=summarize_image_shape(images),
        balance=summarize_class_balance(labels, class_names),
        intensity=summarize_intensity(images, labels, num_classes=len(class_names)),
        class_mean_images=class_mean_images,
        class_similarity=compute_class_similarity(class_mean_images),
    )


def most_similar_class_pairs(
    profile: DatasetProfile,
    *,
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """Return the ``top_k`` most similar distinct class pairs, most similar first.

    These are the pairs a model is expected to confuse, established before any
    training has happened.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    similarity = profile.class_similarity
    names = profile.balance.class_names

    pairs = [
        (names[row], names[column], float(similarity[row, column]))
        for row in range(len(names))
        for column in range(row + 1, len(names))
    ]
    pairs.sort(key=lambda pair: pair[2], reverse=True)
    return pairs[:top_k]


def summary_to_dict(profile: DatasetProfile) -> dict:
    """Return a JSON-ready view of a profile, excluding the image-sized arrays.

    The report cites this file instead of numbers retyped by hand, so a figure in the
    report can never silently disagree with the run that produced it.
    """
    return {
        "dataset": profile.dataset,
        "split": profile.split,
        "shape": {
            "num_samples": profile.shape.num_samples,
            "height": profile.shape.height,
            "width": profile.shape.width,
            "channels": profile.shape.channels,
            "dtype": profile.shape.dtype,
            "min_value": profile.shape.min_value,
            "max_value": profile.shape.max_value,
        },
        "class_balance": {
            "class_names": list(profile.balance.class_names),
            "counts": list(profile.balance.counts),
            "proportions": list(profile.balance.proportions),
            "imbalance_ratio": profile.balance.imbalance_ratio,
            "rarest_class": profile.balance.rarest_class,
            "most_common_class": profile.balance.most_common_class,
        },
        "intensity": {
            "mean": profile.intensity.mean,
            "std": profile.intensity.std,
            "foreground_threshold": FOREGROUND_THRESHOLD,
            "histogram": list(profile.intensity.histogram),
            "bin_edges": list(profile.intensity.bin_edges),
            "per_class_mean_intensity": list(profile.intensity.per_class_mean_intensity),
            "per_class_foreground_fraction": list(profile.intensity.per_class_foreground_fraction),
        },
        "class_similarity": profile.class_similarity.tolist(),
        "most_similar_class_pairs": [
            {"first": first, "second": second, "similarity": similarity}
            for first, second, similarity in most_similar_class_pairs(profile)
        ],
    }


def write_summary_json(profile: DatasetProfile, output_path: str | Path) -> Path:
    """Write :func:`summary_to_dict` to disk and return the path written."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary_to_dict(profile), indent=2) + "\n", encoding="utf-8")
    return path


def format_summary_lines(profile: DatasetProfile) -> list[str]:
    """Render the console summary printed by the EDA entry points."""
    shape = profile.shape
    balance = profile.balance
    intensity = profile.intensity

    lines = [
        f"Dataset: {profile.dataset} ({profile.split} split)",
        f"Number of samples: {shape.num_samples}",
        f"Image size: {shape.height}x{shape.width}, channels: {shape.channels}, "
        f"dtype: {shape.dtype}, value range: [{shape.min_value:.0f}, {shape.max_value:.0f}]",
        "Class distribution:",
    ]
    for class_id, name in enumerate(balance.class_names):
        lines.append(
            f"  class {class_id} ({name}): {balance.counts[class_id]} samples "
            f"({balance.proportions[class_id]:.1%}), "
            f"mean intensity {intensity.per_class_mean_intensity[class_id]:.1f}, "
            f"foreground {intensity.per_class_foreground_fraction[class_id]:.1%}"
        )
    # With perfectly balanced counts argmin and argmax both land on class 0, so naming
    # a "rarest" and "most common" class there would be misleading rather than informative.
    if balance.rarest_class == balance.most_common_class:
        lines.append(
            f"Imbalance ratio (max/min): {balance.imbalance_ratio:.2f} "
            "(every class has equal support)"
        )
    else:
        lines.append(
            f"Imbalance ratio (max/min): {balance.imbalance_ratio:.2f} "
            f"(most common: {balance.most_common_class}, rarest: {balance.rarest_class})"
        )
    lines.append(f"Pixel intensity: mean {intensity.mean:.2f}, std {intensity.std:.2f}")
    lines.append("Most similar class pairs (expected confusions):")
    lines.extend(
        f"  {first} vs {second}: {similarity:+.3f}"
        for first, second, similarity in most_similar_class_pairs(profile)
    )
    return lines
