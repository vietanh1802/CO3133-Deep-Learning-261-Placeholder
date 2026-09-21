"""Run the required Fashion-MNIST exploratory analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import FashionMNIST

from src.data.fashion_mnist import class_names

OUTPUT_DIR = Path("results/a1/eda/fashion_mnist")


def main() -> None:
    """Wire the Fashion-MNIST EDA workflow."""
    dataset = FashionMNIST(root="data", train=True, download=True)
    labels = dataset.targets.numpy()
    names = list(class_names())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image, _ = dataset[0]
    print(f"Number of training samples: {len(dataset)}")
    print(f"Image size: {image.size}, mode: {image.mode}")

    counts = np.bincount(labels, minlength=len(names))
    for class_id, (name, count) in enumerate(zip(names, counts, strict=True)):
        print(f"  class {class_id} ({name}): {count} samples ({count / len(labels):.1%})")
    print(f"Imbalance ratio (max/min): {int(np.max(counts)) / int(np.min(counts)):.2f}")

    _plot_class_distribution(names, counts)
    _plot_representative_samples(dataset, labels, names)


def _plot_class_distribution(names: list[str], counts: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, counts)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.set_title("Fashion-MNIST training class distribution")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png")
    plt.close(fig)


def _plot_representative_samples(
    dataset: FashionMNIST, labels: np.ndarray, names: list[str]
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for class_id, ax in enumerate(axes.flat):
        index = int(np.flatnonzero(labels == class_id)[0])
        image, _ = dataset[index]
        ax.imshow(image, cmap="gray")
        ax.set_title(names[class_id], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "representative_samples.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
