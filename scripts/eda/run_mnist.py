"""Run MNIST exploratory analysis for debugging only."""

from pathlib import Path

from torchvision.datasets import MNIST

from src.data import class_names
from src.data.eda_report import run_dataset_eda

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results/a1/eda/mnist")


def main() -> None:
    """Wire the MNIST EDA workflow."""
    run_dataset_eda(
        train_dataset=MNIST(root=DATA_DIR, train=True, download=True),
        test_dataset=MNIST(root=DATA_DIR, train=False, download=True),
        class_names=class_names("mnist"),
        dataset_name="mnist",
        display_name="MNIST",
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
