import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluation import evaluate
from scripts.training import train_image_classification
from src.config.classification import load_config
from src.data.base import DataLoaders
from src.models import ImageClassifier, build_model
from src.training.checkpoint import load_checkpoint

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs/image_classification/fashion_mnist/lstm.yaml"
)


def test_lstm_train_and_evaluate_entry_points(tmp_path, monkeypatch):
    config = replace(
        load_config(CONFIG_PATH),
        name="lstm_pipeline_smoke",
        epochs=2,
        batch_size=4,
        results_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    config_path = tmp_path / "lstm.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    generator = torch.Generator().manual_seed(17)

    def make_loader(count):
        dataset = TensorDataset(
            torch.randn(count, 1, 28, 28, generator=generator),
            torch.arange(count) % 10,
        )
        return DataLoader(dataset, batch_size=config.batch_size)

    loaders = DataLoaders(train=make_loader(6), validation=make_loader(3), test=make_loader(10))
    monkeypatch.setattr(train_image_classification, "build_dataloaders", lambda *_: loaders)
    monkeypatch.setattr(evaluate, "build_dataloaders", lambda *_: loaders)
    monkeypatch.setattr(train_image_classification, "_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys, "argv", ["train", "--data", "fashion_mnist", "--config", str(config_path)]
    )
    train_image_classification.main()

    run_dir = config.results_dir / "a1" / config.name
    checkpoint = config.checkpoint_dir / config.name / "best.pt"
    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert len(summary["train_loss"]) == len(summary["validation_loss"]) == 2
    restored = ImageClassifier(build_model(config.model, config.model_config))
    metadata = load_checkpoint(checkpoint, restored, restore_rng=False)
    assert metadata.epoch == summary["best_epoch"]
    assert metadata.best_validation_loss == min(summary["validation_loss"])

    monkeypatch.setattr(
        sys, "argv", ["evaluate", "--data", "fashion_mnist", "--config", str(config_path)]
    )
    evaluate.main()

    record = json.loads((run_dir / "test_result.json").read_text(encoding="utf-8"))
    assert record["model"] == "lstm"
    assert record["sample_count"] == 10
    assert record["checkpoint"]["epoch"] == summary["best_epoch"]
    assert record["checkpoint"]["selection_rule"] == "lowest validation loss"
    assert record["parameter_count"] == 82186
    assert 0 <= record["metrics"]["accuracy"] <= 1
    assert 0 <= record["metrics"]["macro_f1"] <= 1
    assert record["training_seconds"] == summary["training_seconds"]
    assert record["inference_seconds"] >= 0
    evidence = torch.load(run_dir / "test_evidence.pt", weights_only=True)
    assert int(evidence["confusion_matrix"].sum()) == 10
    assert evidence["probabilities"].shape == (10, 10)
    for filename in (
        "learning_curves.png",
        "test_confusion_matrix.png",
        "test_error_analysis.json",
        "test_confident_errors.png",
        "test_uncertain_errors.png",
        "test_confident_correct.png",
    ):
        assert (run_dir / filename).is_file()
