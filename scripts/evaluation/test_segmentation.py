"""Score a selected SUIM checkpoint once on the official test split.

Usage:
    python -m scripts.evaluation.test_segmentation --checkpoint checkpoints/<run>/best.pt \
        --final-test

The checkpoint must be the one chosen on validation: its epoch has to match the best
epoch in results/a2/<run>/training_summary.json. The script writes test_summary.json
and test_evidence.pt next to that summary and refuses to overwrite an existing test
summary, so each selected model is scored on the test split only once. Pass the
evidence to eval_segmentation --final-test for the confusion matrix and examples.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.training.train_segmentation import export_evidence
from src.data.suim import CLASS_NAMES, NUM_CLASSES, SuimSegmentation
from src.evaluation.segmentation import evaluate_model
from src.models import build_model
from src.utils.environment import git_revision, software_versions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--final-test", action="store_true", help="confirm configurations are fixed"
    )
    args = parser.parse_args()
    if not args.final_test:
        parser.error("the test split is scored only with --final-test, after selection is fixed")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    config = metadata["config"]
    run_dir = Path(config["results_dir"]) / "a2" / config["name"]
    training = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    if metadata["epoch"] != training["best_epoch"]:
        raise ValueError(
            f"checkpoint epoch {metadata['epoch']} is not the selected epoch "
            f"{training['best_epoch']} of {config['name']}"
        )
    output = run_dir / "test_summary.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; the test split is scored once per model")

    model = build_model(config["model"], config["model_config"])
    model.load_state_dict(payload["model"])
    size = (config["image_height"], config["image_width"])
    target = (config.get("evaluation_height") or size[0], config.get("evaluation_width") or size[1])
    dataset = SuimSegmentation(config["data_dir"], "test", image_size=size, target_size=target)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    regional = evaluate_model(model, loader, device, num_classes=NUM_CLASSES)
    with torch.no_grad():
        export_evidence(model, loader, device, run_dir / "test_evidence.pt", split="test")
    summary = {
        "name": config["name"],
        "split": "test",
        "images": len(dataset),
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": metadata["epoch"],
        "checkpoint_validation_miou": metadata["best_validation_miou"],
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "software": software_versions(),
        "git": git_revision(),
        "test": {
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
        "config": config,
    }
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"{config['name']}: test mIoU {100 * regional['all'].mean_iou:.2f} on {len(dataset)} images"
    )


if __name__ == "__main__":
    main()
