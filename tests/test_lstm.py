import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluation import evaluate
from scripts.training import train_image_classification
from src.data.base import DataLoaders
from src.models import ImageClassifier, build_model
from src.models.lstm import LSTM, LSTMCell
from src.training.checkpoint import load_checkpoint
from src.utils import load_config

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs/image_classification/fashion_mnist/lstm.yaml"
)


@pytest.mark.parametrize("layers", [1, 2, 3])
@pytest.mark.parametrize("batch,length", [(1, 1), (3, 7)])
def test_lstm_matches_reference_outputs_and_gradients(layers, batch, length):
    torch.manual_seed(42)
    model = LSTM(input_size=4, hidden_size=5, num_layers=layers).double()
    reference = nn.LSTM(4, 5, num_layers=layers, batch_first=True).double()
    with torch.no_grad():
        for index, cell in enumerate(model.cells):
            assert isinstance(cell, LSTMCell)
            assert cell.project_input.bias is not None
            assert cell.project_hidden.bias is not None
            for name, parameter in (
                ("weight_ih", cell.project_input.weight),
                ("bias_ih", cell.project_input.bias),
                ("weight_hh", cell.project_hidden.weight),
                ("bias_hh", cell.project_hidden.bias),
            ):
                getattr(reference, f"{name}_l{index}").copy_(parameter)

    images = torch.randn(batch, 1, length, 4, dtype=torch.float64, requires_grad=True)
    sequence = images.detach().squeeze(1).clone().requires_grad_(True)
    features = model(images)
    _, (hidden, _) = reference(sequence)
    torch.testing.assert_close(features, hidden[-1], rtol=1e-9, atol=1e-11)

    probe = torch.randn_like(features)
    (features * probe).sum().backward()
    (hidden[-1] * probe).sum().backward()
    assert images.grad is not None
    assert sequence.grad is not None
    torch.testing.assert_close(images.grad.squeeze(1), sequence.grad, rtol=1e-9, atol=1e-11)
    for index, cell in enumerate(model.cells):
        assert isinstance(cell, LSTMCell)
        assert cell.project_input.bias is not None
        assert cell.project_hidden.bias is not None
        for name, parameter in (
            ("weight_ih", cell.project_input.weight),
            ("bias_ih", cell.project_input.bias),
            ("weight_hh", cell.project_hidden.weight),
            ("bias_hh", cell.project_hidden.bias),
        ):
            torch.testing.assert_close(
                parameter.grad,
                getattr(reference, f"{name}_l{index}").grad,
                rtol=1e-9,
                atol=1e-11,
            )


def test_lstm_config_builds_fashion_mnist_classifier():
    config = load_config(CONFIG_PATH)
    encoder = build_model(config.model, config.model_config)
    assert isinstance(encoder, LSTM)
    model = ImageClassifier(encoder)
    logits = model(torch.randn(3, 1, 28, 28))
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 4, 9]))
    loss.backward()
    assert logits.shape == (3, 10)
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("layers", [1, 2])
def test_lstm_dropout_modes_and_no_state_leak_between_batches(layers):
    torch.manual_seed(17)
    model = LSTM(input_size=4, hidden_size=8, num_layers=layers, dropout=0.5)
    images = torch.randn(3, 1, 6, 4)
    model.eval()
    expected = model(images)
    model(torch.randn_like(images))
    torch.testing.assert_close(model(images), expected, rtol=0, atol=0)
    model.train()
    first, second = model(images), model(images)
    if layers == 1:
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    else:
        assert not torch.equal(first, second)


@pytest.mark.parametrize("shape", [(2, 3, 5, 4), (2, 5, 4), (2, 1, 5, 3), (2, 1, 0, 4)])
def test_lstm_rejects_invalid_image_shapes(shape):
    with pytest.raises(ValueError):
        LSTM(input_size=4)(torch.randn(*shape))


def test_lstm_cell_rejects_mismatched_state():
    with pytest.raises(ValueError, match="hidden and cell must have the same shape"):
        LSTMCell(3, 5)(torch.randn(2, 3), torch.randn(2, 5), torch.randn(2, 4))


def test_lstm_train_and_evaluate_entry_points(tmp_path, monkeypatch):
    # Only replace the data source; use the real config, model, engine, and exporters.
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
    checkpoint = config.checkpoint_dir / "a1.zip"
    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert len(summary["train_loss"]) == len(summary["validation_loss"]) == 2
    restored = ImageClassifier(build_model(config.model, config.model_config))
    metadata = load_checkpoint(checkpoint, restored, restore_rng=False, run_name=config.name)
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
        "test_representative_correct.png",
    ):
        assert (run_dir / filename).is_file()
