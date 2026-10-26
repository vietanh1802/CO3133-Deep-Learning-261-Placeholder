"""Run the required Fashion-MNIST exploratory analysis."""

from pathlib import Path

from torchvision.datasets import FashionMNIST

from src.data.eda_report import run_dataset_eda
from src.data.fashion_mnist import class_names

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results/a1/eda/fashion_mnist")


def main() -> None:
    """Wire the Fashion-MNIST EDA workflow."""
    run_dataset_eda(
        train_dataset=FashionMNIST(root=DATA_DIR, train=True, download=True),
        test_dataset=FashionMNIST(root=DATA_DIR, train=False, download=True),
        class_names=class_names(),
        dataset_name="fashion_mnist",
        display_name="Fashion-MNIST",
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
