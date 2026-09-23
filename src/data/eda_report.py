"""One reusable EDA run, shared by every dataset entry point in ``scripts/eda/``.

Adding a dataset means calling :func:`run_dataset_eda` with its two splits and its
class names; the set of figures and the summary file stay identical across datasets,
which is what makes the report's dataset sections directly comparable.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.data.eda import (
    DatasetProfile,
    format_summary_lines,
    profile_dataset,
    summarize_class_balance,
    write_summary_json,
)
from src.data.eda_plots import (
    plot_class_distribution,
    plot_class_mean_images,
    plot_class_similarity,
    plot_foreground_coverage,
    plot_intensity_histogram,
    plot_representative_samples,
    plot_split_class_distribution,
)


class RawImageDataset(Protocol):
    """The part of a torchvision dataset the EDA needs: undecoded images and labels."""

    data: Any
    targets: Any


def dataset_arrays(dataset: RawImageDataset) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, labels)`` as numpy arrays, without applying any transform.

    ``dataset.data`` is read directly instead of iterating ``dataset[i]``: EDA must
    describe the *raw* data, not the augmented tensors the model sees, and reading the
    backing array is roughly two orders of magnitude faster than decoding 60,000 PIL
    images one at a time.
    """
    images = np.asarray(dataset.data)
    labels = np.asarray(dataset.targets).astype(np.int64).ravel()
    if images.shape[0] != labels.shape[0]:
        raise ValueError("dataset images and targets must have the same length")
    return images, labels


def run_dataset_eda(
    *,
    train_dataset: RawImageDataset,
    test_dataset: RawImageDataset,
    class_names: Sequence[str],
    dataset_name: str,
    display_name: str,
    output_dir: str | Path,
) -> DatasetProfile:
    """Write every EDA figure plus ``summary.json`` for one dataset, and print a recap."""
    output_path = Path(output_dir)
    train_images, train_labels = dataset_arrays(train_dataset)
    test_images, test_labels = dataset_arrays(test_dataset)

    profile = profile_dataset(
        train_images,
        train_labels,
        class_names,
        dataset=dataset_name,
        split="train",
    )

    # Required by the assignment: class distribution, input size, imbalance, samples.
    plot_class_distribution(
        profile.balance,
        output_path / "class_distribution.png",
        title=f"{display_name} training class distribution",
    )
    plot_representative_samples(
        train_images,
        train_labels,
        class_names,
        output_path / "representative_samples.png",
        title=f"{display_name} representative training samples",
    )

    # Supporting analysis: evidence for the preprocessing and error-analysis sections.
    plot_split_class_distribution(
        profile.balance,
        summarize_class_balance(test_labels, class_names),
        output_path / "split_class_distribution.png",
        title=f"{display_name} train vs test class proportions",
    )
    plot_intensity_histogram(
        profile.intensity,
        output_path / "pixel_intensity_distribution.png",
        title=f"{display_name} pixel intensity distribution",
    )
    plot_class_mean_images(
        profile,
        output_path / "class_mean_images.png",
        title=f"{display_name} average image per class",
    )
    plot_class_similarity(
        profile,
        output_path / "class_similarity.png",
        title=f"{display_name} class prototype similarity",
    )
    plot_foreground_coverage(
        profile,
        output_path / "foreground_coverage.png",
        title=f"{display_name} foreground coverage per class",
    )

    summary_path = write_summary_json(profile, output_path / "summary.json")

    for line in format_summary_lines(profile):
        print(line)
    print(f"Figures and summary written to {output_path}")
    print(f"Machine-readable summary: {summary_path}")

    return profile
