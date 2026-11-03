"""Run shared EDA and export comparable dataset summaries and figures."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.eda.base import RawImageDataset
from src.eda.classification import (
    DatasetProfile,
    format_summary_lines,
    profile_dataset,
    summarize_class_balance,
    write_summary_json,
)
from src.plots.eda import (
    plot_class_distribution,
    plot_class_mean_images,
    plot_class_similarity,
    plot_foreground_coverage,
    plot_intensity_histogram,
    plot_representative_samples,
    plot_split_class_distribution,
)


def dataset_arrays(dataset: RawImageDataset) -> tuple[np.ndarray, np.ndarray]:
    """Read raw dataset arrays without transforms.

    Returns:
        (images, labels) NumPy arrays read directly from dataset storage.
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

    # Assignment EDA outputs.
    plot_class_distribution(
        profile.balance,
        output_path / "class_distribution.png",
        title=f"{display_name} training class counts",
    )
    plot_representative_samples(
        train_images,
        train_labels,
        class_names,
        output_path / "representative_samples.png",
        title=f"{display_name} training examples",
    )

    # Preprocessing and error-analysis evidence.
    plot_split_class_distribution(
        profile.balance,
        summarize_class_balance(test_labels, class_names),
        output_path / "split_class_distribution.png",
        title=f"{display_name} split proportions",
    )
    plot_intensity_histogram(
        profile.intensity,
        output_path / "pixel_intensity_distribution.png",
        title=f"{display_name} pixel intensity",
    )
    plot_class_mean_images(
        profile,
        output_path / "class_mean_images.png",
        title=f"{display_name} class means",
    )
    plot_class_similarity(
        profile,
        output_path / "class_similarity.png",
        title=f"{display_name} prototype similarity",
    )
    plot_foreground_coverage(
        profile,
        output_path / "foreground_coverage.png",
        title=f"{display_name} foreground coverage",
    )

    summary_path = write_summary_json(profile, output_path / "summary.json")

    for line in format_summary_lines(profile):
        print(line)
    print(f"Figures and summary written to {output_path}")
    print(f"Machine-readable summary: {summary_path}")

    return profile
