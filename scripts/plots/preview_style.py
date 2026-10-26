"""Render five deterministic synthetic style previews without fitting any model.

Run with python -m scripts.plots.preview_style --output-dir /tmp/co3133-plot-preview.
All plotted values are illustrative, not experiment results.
"""

import argparse
from pathlib import Path

import numpy as np

from src.eda.classification import summarize_class_balance
from src.plots.classification import plot_learning_curves
from src.plots.common import plot_confusion_matrix
from src.plots.eda import plot_representative_samples, plot_split_class_distribution
from src.plots.segmentation import plot_segmentation_curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir
    training = [1.5, 0.9, 0.65, 0.45, 0.32]
    validation = [1.55, 1.0, 0.82, 0.88, 0.91]
    plot_learning_curves(training, validation, output / "classification_curves.png")
    plot_segmentation_curves(
        training, validation, [0.1, 0.25, 0.4, 0.38, 0.37], output / "segmentation_curves.png"
    )
    names = ("Circle", "Rectangle", "Stripe")
    plot_confusion_matrix(
        np.array([[18, 2, 0], [1, 16, 3], [0, 2, 18]]), names, output / "confusion_matrix.png"
    )
    labels = np.repeat(np.arange(3), 4)
    plot_split_class_distribution(
        summarize_class_balance(labels, names),
        summarize_class_balance(np.repeat(np.arange(3), [5, 4, 3]), names),
        output / "eda_comparison.png",
        title="Class proportions",
    )
    rng = np.random.default_rng(42)
    images = rng.integers(0, 30, (12, 16, 16), dtype=np.uint8)
    yy, xx = np.mgrid[:16, :16]
    for index, label in enumerate(labels):
        shift = index % 4 - 1
        if label == 0:
            shape = (xx - 8 - shift) ** 2 + (yy - 8) ** 2 < 20
        elif label == 1:
            shape = (xx >= 5 + shift) & (xx <= 10 + shift) & (yy >= 3) & (yy <= 12)
        else:
            shape = (yy >= 6 + shift) & (yy <= 9 + shift)
        images[index, shape] = 220
    plot_representative_samples(
        images, labels, names, output / "image_grid.png", title="Synthetic examples"
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
