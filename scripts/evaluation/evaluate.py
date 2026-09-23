"""Evaluate a selected checkpoint on the held-out test split."""

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from src.data import build_dataloaders, class_names
from src.error import ClassificationError, ClassificationErrorAnalyzer
from src.evaluation import evaluate_model
from src.evaluation.plots import plot_confusion_matrix, plot_prediction_examples
from src.models import ImageClassifier, build_model
from src.training.checkpoint import load_checkpoint
from src.utils import ExperimentState, load_config, seed_everything, setup_logging
from src.utils.logging import log_config, log_timing


def _git_commit() -> str:
    """Return the current commit hash, marked dirty when the tree has changes."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    revision = completed.stdout.strip()
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return revision
    return f"{revision}-dirty" if dirty else revision


def _environment() -> dict[str, Any]:
    """Capture the hardware and software versions reported with each result."""
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
    )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device_name": device_name or "unknown",
    }


def _plot_examples(
    examples: tuple[ClassificationError, ...],
    dataset: Any,
    names: tuple[str, ...],
    output_path: Path,
    title: str,
) -> None:
    """Load selected dataset samples and write one qualitative-analysis figure."""
    images = [dataset[example.sample_index][0].detach().cpu().numpy() for example in examples]
    plot_prediction_examples(
        images,
        [example.target for example in examples],
        [example.prediction for example in examples],
        [example.confidence for example in examples],
        names,
        output_path,
        title=title,
    )


def parse_args() -> argparse.Namespace:
    """Parse evaluation command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        choices=("fashion_mnist", "mnist", "cifar10"),
        help="dataset to evaluate",
    )
    parser.add_argument("--config", required=True, help="experiment configuration YAML")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint to evaluate (default: <checkpoint_dir>/<name>/best.pt)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("test", "validation"),
        help="split to evaluate; reported results must come from the test split",
    )
    return parser.parse_args()


def main() -> None:
    """Wire checkpoint loading, evaluation, and result persistence."""
    args = parse_args()
    config = replace(load_config(args.config), dataset=args.data)

    run_directory = Path(config.results_dir) / "a1" / config.name
    run_directory.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=run_directory / "evaluate.log")
    logger.info(ExperimentState.INITIALIZING)
    log_config(config.to_dict())

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    dataloader = getattr(dataloaders, args.split)

    encoder = build_model(config.model, config.model_config)
    model = ImageClassifier(encoder)

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(config.checkpoint_dir) / config.name / "best.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    metadata = load_checkpoint(
        checkpoint_path,
        model,
        map_location=device,
        restore_rng=False,
    )
    logger.info(
        "Loaded checkpoint: {} (epoch {}, val loss {:.4f})",
        checkpoint_path,
        metadata.epoch,
        metadata.best_validation_loss,
    )

    logger.info(ExperimentState.TESTING)
    with log_timing(f"evaluate {config.name} on {args.split}"):
        result = evaluate_model(model, dataloader, device)

    metrics = result.metrics
    logger.info("Accuracy: {:.4f}", metrics.accuracy)
    logger.info("Macro-F1: {:.4f}", metrics.macro_f1)
    logger.info("Parameters: {}", result.parameter_count)
    logger.info("Inference seconds: {:.4f}", result.inference_seconds)

    logger.info(ExperimentState.SAVING)

    training_summary_path = run_directory / "training_summary.json"
    training_seconds = None
    if training_summary_path.exists():
        training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
        training_seconds = training_summary.get("training_seconds")

    record: dict[str, Any] = {
        "name": config.name,
        "model": config.model,
        "dataset": config.dataset,
        "split": args.split,
        "seed": config.seed,
        "config": config.to_dict(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "selection_rule": "lowest validation loss",
            **asdict(metadata),
        },
        "metrics": {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
        },
        "parameter_count": result.parameter_count,
        "training_seconds": training_seconds,
        "inference_seconds": result.inference_seconds,
        "sample_count": int(result.targets.numel()),
        "git_commit": _git_commit(),
        "environment": _environment(),
    }
    result_path = run_directory / f"{args.split}_result.json"
    result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("Wrote {}", result_path)

    evidence_path = run_directory / f"{args.split}_evidence.pt"
    torch.save(
        {
            "sample_indices": result.sample_indices,
            "targets": result.targets,
            "predictions": result.predictions,
            "probabilities": result.probabilities,
            "confusion_matrix": metrics.confusion_matrix,
        },
        evidence_path,
    )
    logger.info("Wrote {}", evidence_path)

    names = class_names(config.dataset)
    plot_confusion_matrix(
        metrics.confusion_matrix.numpy(),
        names,
        run_directory / f"{args.split}_confusion_matrix.png",
    )

    error_report = ClassificationErrorAnalyzer().analyze(result)
    error_record = {
        "per_class_accuracy": {
            name: float(accuracy)
            for name, accuracy in zip(names, error_report.per_class_accuracy, strict=True)
        },
        "per_class_support": {
            name: int(support)
            for name, support in zip(names, error_report.per_class_support, strict=True)
        },
        "most_confused_pairs": [asdict(pair) for pair in error_report.most_confused_pairs],
        "confident_errors": [asdict(example) for example in error_report.confident_errors],
        "uncertain_errors": [asdict(example) for example in error_report.uncertain_errors],
        "representative_correct": [
            asdict(example) for example in error_report.representative_correct
        ],
    }
    error_path = run_directory / f"{args.split}_error_analysis.json"
    error_path.write_text(json.dumps(error_record, indent=2), encoding="utf-8")
    logger.info("Wrote {}", error_path)

    for examples, filename, title in (
        (
            error_report.representative_correct,
            "representative_correct.png",
            "Representative correct predictions",
        ),
        (error_report.confident_errors, "confident_errors.png", "Confident errors"),
        (error_report.uncertain_errors, "uncertain_errors.png", "Uncertain errors"),
    ):
        _plot_examples(
            examples,
            dataloader.dataset,
            names,
            run_directory / f"{args.split}_{filename}",
            title,
        )
    logger.success(ExperimentState.COMPLETED)


if __name__ == "__main__":
    main()
