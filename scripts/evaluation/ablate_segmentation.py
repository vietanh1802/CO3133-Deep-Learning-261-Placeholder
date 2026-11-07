"""Plan or run five one-factor comparisons against one shared DeepLabV3 reference."""

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from src.config.segmentation import SegmentationConfig, load_segmentation_config
from src.data.suim import CLASS_NAMES


def ablation_configs(base: SegmentationConfig, seeds: list[int]) -> list[SegmentationConfig]:
    """Create six configurations per seed, changing one experimental factor each."""
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("seeds must be distinct integers in [0, 2**32)")
    if (
        base.model != "deeplabv3"
        or base.dataset != "suim"
        or base.model_config.get("num_classes", 8) != 8
        or base.model_config.get("backbone", "resnet50") != "resnet50"
        or base.model_config.get("freeze_backbone", False)
        or not base.model_config.get("freeze_batch_norm", True)
        or not base.augment
        or base.loss != "cross_entropy"
        or min(base.image_height, base.image_width) < 2
    ):
        raise ValueError("reference requires full ResNet50 fine-tuning, fixed BN, flips and CE")
    configs = []
    for seed in seeds:
        reference = replace(
            base,
            name=f"suim_ablation_reference_seed{seed}",
            seed=seed,
            model_config={**base.model_config, "backbone": "resnet50", "freeze_backbone": False},
            evaluation_height=base.image_height,
            evaluation_width=base.image_width,
        )
        configs.append(reference)
        for variant, changes in (
            ("frozen", {"model_config": {**reference.model_config, "freeze_backbone": True}}),
            ("no_augmentation", {"augment": False}),
            ("resnet101", {"model_config": {**reference.model_config, "backbone": "resnet101"}}),
            ("ce_dice", {"loss": "ce_dice"}),
            (
                "low_resolution",
                {"image_height": base.image_height // 2, "image_width": base.image_width // 2},
            ),
        ):
            configs.append(
                replace(reference, name=f"suim_ablation_{variant}_seed{seed}", **changes)
            )
    return configs


def write_comparisons(configs: list[SegmentationConfig], output: Path) -> None:
    """Export paired validation results, rejecting incomplete or mismatched runs."""
    summaries = {}
    for config in configs:
        path = config.results_dir / "a2" / config.name / "training_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["config"] != config.to_dict() or len(summary["history"]) != config.epochs:
            raise ValueError(f"incomplete or mismatched result: {path}")
        summaries[config.name] = summary
    rows = []
    for config in configs:
        run = summaries[config.name]
        reference = summaries[f"suim_ablation_reference_seed{config.seed}"]
        for key in ("device", "software", "world_size", "effective_batch_size", "git"):
            if run[key] != reference[key]:
                raise ValueError(f"{config.name}: {key} differs from the reference")
        scores = run["validation"]["all"]
        reference_scores = reference["validation"]["all"]
        rows.append(
            {
                "run": config.name,
                "seed": config.seed,
                "best_epoch": run["best_epoch"],
                "miou": scores["mean_iou"],
                "mean_dice": scores["mean_dice"],
                "delta_miou_pp": 100 * (scores["mean_iou"] - reference_scores["mean_iou"]),
                "delta_dice_pp": 100 * (scores["mean_dice"] - reference_scores["mean_dice"]),
                **{f"iou_{name}": scores["per_class_iou"][name] for name in CLASS_NAMES},
                "boundary_miou": run["validation"]["boundary"]["mean_iou"],
                "parameters": run["parameters"],
                "trainable_parameters": run["trainable_parameters"],
                "training_seconds": run["training_seconds"],
                "peak_training_memory_mb": run["peak_training_memory_mb"],
                "inference_ms_per_image": run["validation_inference"]["milliseconds_per_image"],
            }
        )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Generate a reviewable plan; launch sequential single-process runs with --run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/segmentation/suim/ablation.yaml")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--run", action="store_true", help="execute the generated configurations")
    parser.add_argument("--resume", action="store_true", help="continue existing runs with --run")
    args = parser.parse_args()
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    base = load_segmentation_config(args.config)
    configs = ablation_configs(base, args.seeds)
    directory = base.results_dir / "a2" / "ablation"
    manifest_path = directory / "manifest.json"
    manifest = {config.name: config.to_dict() for config in configs}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("existing study settings differ; use a separate results_dir")
    # Validate every destination before launching any run or changing the saved plan.
    for config in configs:
        path = directory / "configs" / f"{config.name}.yaml"
        if path.exists() and load_segmentation_config(path) != config:
            raise ValueError(f"existing configuration differs: {path}")
        run_dir = config.results_dir / "a2" / config.name
        checkpoints = config.checkpoint_dir / config.name
        if args.run and (run_dir.exists() or checkpoints.exists()):
            if not args.resume:
                raise FileExistsError(f"{config.name} exists; use --run --resume")
            if not (checkpoints / "last.pt").is_file():
                raise FileNotFoundError(f"cannot resume {config.name}: last.pt is missing")
            checkpoint = torch.load(checkpoints / "last.pt", map_location="cpu", weights_only=True)
            if checkpoint["metadata"]["config"] != config.to_dict():
                raise ValueError(f"checkpoint configuration differs: {config.name}")
            if not (checkpoints / "best.pt").is_file():
                raise FileNotFoundError(f"cannot resume {config.name}: best.pt is missing")
            del checkpoint
            summary_path = run_dir / "training_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary["config"] != config.to_dict():
                    raise ValueError(f"summary configuration differs: {config.name}")
    (directory / "configs").mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for config in configs:
        path = directory / "configs" / f"{config.name}.yaml"
        path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "scripts.training.train_segmentation",
            "--config",
            str(path),
        ]
        if args.resume and (config.checkpoint_dir / config.name / "last.pt").is_file():
            command.append("--resume")
        print(" ".join(command), flush=True)
        if args.run:
            summary_path = config.results_dir / "a2" / config.name / "training_summary.json"
            if args.resume and summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if len(summary["history"]) == config.epochs:
                    print(f"Already complete: {config.name}", flush=True)
                    continue
            subprocess.run(command, check=True)
    if args.run:
        write_comparisons(configs, directory / "comparisons.csv")


if __name__ == "__main__":
    main()
