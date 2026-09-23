"""Train one configured image classifier."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from loguru import logger
from torch import nn

from src.data import build_dataloaders
from src.evaluation.plots import plot_learning_curves
from src.models import ImageClassifier, build_model
from src.training.engine import fit
from src.utils import ExperimentState, load_config, seed_everything, setup_logging
from src.utils.logging import log_config


def parse_args() -> argparse.Namespace:
    """Parse the dataset and experiment configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        choices=("fashion_mnist", "mnist", "cifar10"),
        help="dataset to train on",
    )
    parser.add_argument("--config", required=True, help="experiment configuration YAML")
    return parser.parse_args()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    """Train, validate, checkpoint, and save the learning curves."""
    args = parse_args()
    config = replace(load_config(args.config), dataset=args.data)

    run_directory = Path(config.results_dir) / "a1" / config.name
    checkpoint_path = Path(config.checkpoint_dir) / config.name / "best.pt"
    run_directory.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=run_directory / "train.log")
    logger.info(ExperimentState.INITIALIZING)
    log_config(config.to_dict())

    seed_everything(config.seed)
    device = _device()
    logger.info("Device: {}", device)

    logger.info(ExperimentState.LOADING_DATA)
    dataloaders = build_dataloaders(
        config.dataset,
        {
            "data_dir": config.data_dir,
            "batch_size": config.batch_size,
            "val_fraction": config.val_fraction,
            "seed": config.seed,
            "num_workers": config.num_workers,
        },
    )

    encoder = build_model(config.model, config.model_config)
    model = ImageClassifier(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    result = fit(
        model,
        dataloaders.train,
        dataloaders.validation,
        optimizer,
        loss_fn,
        device,
        epochs=config.epochs,
        checkpoint_path=checkpoint_path,
        checkpoint_config=config.to_dict(),
    )
    train_losses = [epoch.loss for epoch in result.training]
    validation_losses = [epoch.loss for epoch in result.validation]
    plot_learning_curves(
        train_losses,
        validation_losses,
        run_directory / "learning_curves.png",
    )
    summary = {
        "name": config.name,
        "model": config.model,
        "dataset": config.dataset,
        "seed": config.seed,
        "epochs": config.epochs,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "training_seconds": result.duration_seconds,
        "train_loss": train_losses,
        "validation_loss": validation_losses,
        "checkpoint": str(checkpoint_path),
        "config": config.to_dict(),
    }
    (run_directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.success(
        "{} | best epoch {}, validation loss {:.4f}, {:.1f}s",
        ExperimentState.COMPLETED,
        result.best_epoch,
        result.best_validation_loss,
        result.duration_seconds,
    )


if __name__ == "__main__":
    main()
