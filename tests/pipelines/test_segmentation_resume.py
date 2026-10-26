import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.segmentation import evaluate_model
from src.models import UNet
from src.plots.segmentation import plot_segmentation_curves
from src.training.segmentation import fit_segmentation, load_training_state, warmup_cosine
from src.utils.seed import make_generator, seed_everything

DEVICE = torch.device("cpu")


def test_warmup_cosine_schedule():
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = warmup_cosine(optimizer, warmup_steps=2, total_steps=6)
    rates = []
    for _ in range(6):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[:3] == pytest.approx([0.5, 1.0, 1.0])
    assert rates == sorted(rates[:2]) + sorted(rates[2:], reverse=True)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def _toy_run(directory, epochs, resume=False):
    seed_everything(3)
    images = torch.randn(9, 3, 16, 16)
    targets = torch.randint(0, 3, (9, 16, 16))
    train = DataLoader(
        TensorDataset(images, targets), batch_size=3, shuffle=True, generator=make_generator(3)
    )
    validation = DataLoader(TensorDataset(images[:4], targets[:4]), batch_size=2)
    model = UNet(num_classes=3, base_width=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = warmup_cosine(optimizer, 2, 4)
    return fit_segmentation(
        model,
        train,
        validation,
        optimizer,
        scheduler,
        nn.CrossEntropyLoss(),
        DEVICE,
        epochs=epochs,
        accumulation_steps=2,
        seed=3,
        checkpoint_dir=directory / "checkpoints",
        run_dir=directory / "run",
        checkpoint_config={"name": "toy"},
        num_classes=3,
        resume=resume,
    )


def test_fit_saves_selection_and_resumes_exactly(tmp_path):
    straight = _toy_run(tmp_path / "straight", epochs=2)
    _toy_run(tmp_path / "resumed", epochs=1)
    resumed = _toy_run(tmp_path / "resumed", epochs=2, resume=True)

    for name in ("best.pt", "last.pt"):
        assert (tmp_path / "straight" / "checkpoints" / name).is_file()
    history = json.loads((tmp_path / "straight" / "run" / "history.json").read_text())
    assert [record["epoch"] for record in history] == [1, 2]
    assert straight.best_validation_miou == max(r["validation_miou"] for r in history)
    for left, right in zip(straight.history, resumed.history, strict=True):
        assert left.train_loss == pytest.approx(right.train_loss)
        assert left.validation_miou == pytest.approx(right.validation_miou)

    selected = UNet(num_classes=3, base_width=4)
    checkpoint = tmp_path / "straight/checkpoints/best.pt"
    metadata = load_training_state(checkpoint, selected, restore_rng=False)
    assert metadata.epoch == straight.best_epoch
    images = torch.randn(5, 3, 16, 16)
    targets = torch.randint(0, 3, (5, 16, 16))
    held_out = DataLoader(TensorDataset(images, targets), batch_size=2)
    expected = evaluate_model(selected, held_out, DEVICE, num_classes=3)
    reloaded = UNet(num_classes=3, base_width=4)
    load_training_state(checkpoint, reloaded, restore_rng=False)
    actual = evaluate_model(reloaded, held_out, DEVICE, num_classes=3)
    for region in expected:
        torch.testing.assert_close(
            actual[region].confusion_matrix, expected[region].confusion_matrix, rtol=0, atol=0
        )
    assert actual["all"].confusion_matrix.sum().item() == targets.numel()
    output = tmp_path / "curves.png"
    plot_segmentation_curves(
        [record.train_loss for record in straight.history],
        [record.validation_loss for record in straight.history],
        [record.validation_miou for record in straight.history],
        output,
    )
    assert output.stat().st_size > 0
    assert output.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
