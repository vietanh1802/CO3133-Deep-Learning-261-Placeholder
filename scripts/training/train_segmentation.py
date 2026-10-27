"""Train one configured SUIM segmentation model and export its validation evidence."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from loguru import logger
from torch import distributed as dist
from torch import nn
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel

from src.config.segmentation import load_segmentation_config
from src.data import build_dataloaders
from src.data.segmentation import SegmentationDataset
from src.data.suim import CLASS_NAMES, NUM_CLASSES
from src.evaluation.compute_cost import measure_inference
from src.evaluation.segmentation import evaluate_model
from src.models import build_model
from src.plots.segmentation import (
    plot_segmentation_curves,
)
from src.training.segmentation import (
    build_segmentation_optimizer,
    fit_segmentation,
    load_training_state,
    optimizer_steps_per_epoch,
    warmup_cosine,
)
from src.utils import ExperimentState, seed_everything, setup_logging
from src.utils.distributed import (
    cleanup,
    initialize,
    is_primary,
    rank,
    unwrap,
    world_size,
)
from src.utils.environment import git_revision, software_versions
from src.utils.logging import log_config


def parse_args() -> argparse.Namespace:
    """Parse the experiment configuration and resume flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="segmentation configuration YAML")
    parser.add_argument(
        "--resume", action="store_true", help="continue from the run's last.pt checkpoint"
    )
    return parser.parse_args()


@torch.no_grad()
def export_evidence(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, path: Path
) -> None:
    """Save validation targets, predictions, and display images for eval_segmentation."""
    dataset = dataloader.dataset
    if not isinstance(dataset, SegmentationDataset):
        raise TypeError("evidence export expects a SegmentationDataset")
    model.eval()
    targets, predictions, images = [], [], []
    for batch, target in dataloader:
        output = model(batch.to(device))
        logits = output["out"] if isinstance(output, dict) else output
        predictions.append(logits.argmax(1).to(torch.uint8).cpu())
        targets.append(target.to(torch.uint8))
        images.append(dataset.denormalize(batch))
    torch.save(
        {
            "split": "validation",
            "targets": torch.cat(targets),
            "predictions": torch.cat(predictions),
            "images": torch.cat(images),
            "sample_ids": dataset.stems,
        },
        path,
    )


def run() -> None:
    """Train, select by validation mIoU, re-evaluate the best checkpoint, and save results."""
    args = parse_args()
    config = load_segmentation_config(args.config)
    # Capture tree state before logs mark it dirty.
    revision = git_revision()

    run_directory = Path(config.results_dir) / "a2" / config.name
    checkpoint_directory = Path(config.checkpoint_dir) / config.name
    device = initialize()
    run_directory.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=run_directory / ("train.log" if is_primary() else f"rank{rank()}.log"))
    if not is_primary():
        logger.remove()
        logger.add(run_directory / f"rank{rank()}.log", level="WARNING")
        logger.add(
            lambda message: print(message, end=""),
            level="WARNING",
            format="rank {extra[rank]} | {level} | {message}",
        )
        logger.configure(extra={"rank": rank()})
    logger.info(ExperimentState.INITIALIZING)
    log_config(config.to_dict())

    seed_everything(config.seed)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    logger.info("Device: {}", device_name)

    logger.info(ExperimentState.LOADING_DATA)
    dataloaders = build_dataloaders(
        config.dataset,
        {
            "data_dir": config.data_dir,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "num_workers": config.num_workers,
            "image_height": config.image_height,
            "image_width": config.image_width,
        },
    )

    model = build_model(config.model, config.model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if dist.is_initialized():
        model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    logger.info("Parameters: {:,}", parameter_count)
    logger.info(
        "Processes: {} | per-device batch: {} | effective batch: {}",
        world_size(),
        config.batch_size,
        config.batch_size * config.accumulation_steps * world_size(),
    )
    if dist.is_initialized():
        seed_everything(config.seed + rank())
    optimizer = build_segmentation_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        backbone_learning_rate=config.backbone_learning_rate,
    )
    steps_per_epoch = optimizer_steps_per_epoch(len(dataloaders.train), config.accumulation_steps)
    scheduler = warmup_cosine(
        optimizer, config.warmup_epochs * steps_per_epoch, config.epochs * steps_per_epoch
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    result = fit_segmentation(
        model,
        dataloaders.train,
        dataloaders.validation,
        optimizer,
        scheduler,
        nn.CrossEntropyLoss(),
        device,
        epochs=config.epochs,
        accumulation_steps=config.accumulation_steps,
        seed=config.seed,
        checkpoint_dir=checkpoint_directory,
        run_dir=run_directory,
        checkpoint_config=config.to_dict(),
        num_classes=NUM_CLASSES,
        resume=args.resume,
    )

    # Resume timing covers only this process.
    peak_training_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    )

    if not is_primary():
        return
    model = unwrap(model)
    logger.info(ExperimentState.SAVING)
    load_training_state(checkpoint_directory / "best.pt", model, restore_rng=False)
    regional = evaluate_model(model, dataloaders.validation, device, num_classes=NUM_CLASSES)
    export_evidence(model, dataloaders.validation, device, run_directory / "val_evidence.pt")
    inference = measure_inference(model, dataloaders.validation, device)
    logger.info(
        "Inference: {:.2f} ms/image, {:.1f} images/s at batch {}",
        inference.milliseconds_per_image,
        inference.images_per_second,
        inference.batch_size,
    )
    plot_segmentation_curves(
        [record.train_loss for record in result.history],
        [record.validation_loss for record in result.history],
        [record.validation_miou for record in result.history],
        run_directory / "learning_curves.png",
    )
    summary = {
        "name": config.name,
        "model": config.model,
        "parameters": parameter_count,
        "device": device_name,
        "software": software_versions(),
        "git": revision,
        "seed": config.seed,
        "mixed_precision": False,
        "world_size": world_size(),
        "batch_size_per_device": config.batch_size,
        "effective_batch_size": config.batch_size * config.accumulation_steps * world_size(),
        "checkpoint_selection": "highest validation mIoU",
        "best_epoch": result.best_epoch,
        "best_validation_miou": result.best_validation_miou,
        # Epoch time includes validation.
        "training_seconds": result.duration_seconds,
        "peak_training_memory_mb": peak_training_memory_mb,
        "validation_inference": asdict(inference),
        "checkpoint": str(checkpoint_directory / "best.pt"),
        "validation": {
            region: {
                "mean_iou": metrics.mean_iou,
                "mean_dice": metrics.mean_dice,
                "per_class_iou": dict(
                    zip(CLASS_NAMES, metrics.per_class_iou.tolist(), strict=True)
                ),
                "per_class_dice": dict(
                    zip(CLASS_NAMES, metrics.per_class_dice.tolist(), strict=True)
                ),
            }
            for region, metrics in regional.items()
        },
        "history": [asdict(record) for record in result.history],
        "config": config.to_dict(),
    }
    (run_directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.success(
        "{} | best epoch {}, validation mIoU {:.4f}, {:.1f} min",
        ExperimentState.COMPLETED,
        result.best_epoch,
        result.best_validation_miou,
        result.duration_seconds / 60,
    )


@record
def main() -> None:
    try:
        run()
    except Exception:
        logger.exception("Training failed on rank {}", rank())
        raise
    finally:
        cleanup()


if __name__ == "__main__":
    main()
