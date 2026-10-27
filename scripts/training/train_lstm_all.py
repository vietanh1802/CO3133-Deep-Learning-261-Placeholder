"""Train and test the three A1 LSTM runs with the existing classification pipeline."""

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch
import torchvision

from src.config.classification import load_config

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("mnist", "fashion_mnist", "cifar10")
SHARED_SETTINGS = ("seed", "batch_size", "learning_rate", "epochs", "num_workers", "val_fraction")


def main() -> None:
    """Run each dataset in a separate process; stop immediately on a failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--check-only", action="store_true", help="check GPU and configs without training"
    )
    args = parser.parse_args()
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("each dataset may only be selected once")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is available. Use the previous NVIDIA training environment."
        )

    plans = []
    for dataset in args.datasets:
        config_path = Path("configs/image_classification") / dataset / "lstm.yaml"
        config = replace(load_config(ROOT / config_path), dataset=dataset)
        baseline = json.loads(
            (ROOT / "results/a1" / f"{dataset}_gru_seed42" / "training_summary.json").read_text()
        )["config"]
        for key in SHARED_SETTINGS:
            if getattr(config, key) != baseline[key]:
                raise ValueError(f"{dataset}: {key} differs from the previous GRU run")
        for key in ("hidden_size", "num_layers", "dropout", "input_size"):
            if config.model_config[key] != baseline["model_config"][key]:
                raise ValueError(
                    f"{dataset}: model setting {key} differs from the previous GRU run"
                )
        directory = ROOT / config.results_dir / "a1" / config.name
        checkpoint = ROOT / config.checkpoint_dir / config.name / "best.pt"
        if not args.check_only and any(
            p.exists()
            for p in (
                directory / "training_summary.json",
                directory / "test_result.json",
                checkpoint,
            )
        ):
            raise FileExistsError(
                f"Existing LSTM output for {dataset}. Back it up first, "
                "or use --datasets for other runs."
            )
        plans.append((dataset, config_path, config, directory))

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    environment = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": commit,
        "source_dirty_before_training": bool(status),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "platform": platform.platform(),
        "training_device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "evaluation_device": "cuda",
    }
    print(json.dumps(environment, indent=2), flush=True)
    for dataset, _, config, _ in plans:
        settings = {key: getattr(config, key) for key in SHARED_SETTINGS}
        print(f"{dataset}: {settings}; model={config.model_config}", flush=True)
    if args.check_only:
        return

    for index, (dataset, config_path, config, directory) in enumerate(plans, start=1):
        print(f"\n===== LSTM {index}/{len(plans)}: {dataset} =====", flush=True)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run_environment.json").write_text(
            json.dumps(environment, indent=2) + "\n", encoding="utf-8"
        )
        for module in (
            "scripts.training.train_image_classification",
            "scripts.evaluation.evaluate",
        ):
            print(module, flush=True)
            subprocess.run(
                [sys.executable, "-m", module, "--data", dataset, "--config", str(config_path)],
                cwd=ROOT,
                check=True,
            )
        summary = json.loads((directory / "training_summary.json").read_text())
        record = json.loads((directory / "test_result.json").read_text())
        if (
            len(summary["train_loss"]) != config.epochs
            or len(summary["validation_loss"]) != config.epochs
            or record["split"] != "test"
            or record["sample_count"] != 10000
            or record["checkpoint"]["epoch"] != summary["best_epoch"]
        ):
            raise RuntimeError(f"Incomplete or inconsistent outputs for {dataset}")
        print(f"Completed {dataset}: {record['metrics']}", flush=True)
    print("LSTM train/test completed. Results: results/a1/*_lstm_seed42/", flush=True)


if __name__ == "__main__":
    main()
