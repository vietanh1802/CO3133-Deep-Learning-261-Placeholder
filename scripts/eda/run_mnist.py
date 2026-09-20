"""Run MNIST exploratory analysis for debugging only."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST

OUTPUT_DIR = Path("results/a1/eda/mnist")


def main() -> None:
    """Wire the MNIST EDA workflow."""
    dataset = MNIST(root="data", train=True, download=True)
    labels = dataset.targets.numpy()
    class_ids = sorted(int(c) for c in set(labels.tolist()))
    names = [str(c) for c in class_ids]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image, _ = dataset[0]
    print(f"Number of training samples: {len(dataset)}")
    print(f"Image size: {image.size}, mode: {image.mode}")

    counts = np.bincount(labels, minlength=len(class_ids))
    for class_id, count in zip(class_ids, counts, strict=True):
        print(f"  class {class_id}: {count} samples ({count / len(labels):.1%})")
    print(f"Imbalance ratio (max/min): {int(np.max(counts)) / int(np.min(counts)):.2f}")

    _plot_class_distribution(names, counts)
    _plot_representative_samples(dataset, labels, class_ids, names)


def _plot_class_distribution(names: list[str], counts: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, counts)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.set_title("MNIST training class distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png")
    plt.close(fig)


def _plot_representative_samples(
    dataset: MNIST, labels: np.ndarray, class_ids: list[int], names: list[str]
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for class_id, ax in zip(class_ids, axes.flat, strict=True):
        index = int(np.flatnonzero(labels == class_id)[0])
        image, _ = dataset[index]
        ax.imshow(image, cmap="gray")
        ax.set_title(names[class_id])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "representative_samples.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
