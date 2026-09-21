import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from src.training.engine import EpochResult, eval_one_epoch, train_one_epoch
from src.utils.seed import seed_everything

DEVICE = torch.device("cpu")


def test_checkpoint_round_trip_restores_training_and_rng_state(tmp_path):
    seed_everything(11)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()

    metadata = CheckpointMetadata(
        epoch=3,
        global_step=17,
        best_validation_loss=0.25,
        config={"model": "linear"},
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, metadata)

    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    expected_random = (random.random(), np.random.random(), torch.rand(1))

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    random.random()
    np.random.random()
    torch.rand(1)

    restored_metadata = load_checkpoint(path, model, optimizer)
    actual_random = (random.random(), np.random.random(), torch.rand(1))

    assert restored_metadata == metadata
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(model.parameters(), expected_parameters, strict=True)
    )
    assert actual_random[0] == expected_random[0]
    assert actual_random[1] == expected_random[1]
    assert torch.equal(actual_random[2], expected_random[2])


def build_samples(sample_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw the same features and targets on every call."""
    seed_everything(0)
    return torch.randn(sample_count, 4), torch.randint(0, 3, (sample_count,))


def build_dataloader(sample_count: int, batch_size: int) -> DataLoader:
    """Build a deterministic loader whose last batch is deliberately uneven."""
    features, targets = build_samples(sample_count)
    return DataLoader(TensorDataset(features, targets), batch_size=batch_size, shuffle=False)


def test_train_one_epoch_updates_parameters_and_counts_every_sample() -> None:
    seed_everything(0)
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    dataloader = build_dataloader(5, 2)

    result = train_one_epoch(model, dataloader, optimizer, nn.CrossEntropyLoss(), DEVICE)

    assert isinstance(result, EpochResult)
    assert result.sample_count == 5
    assert 0.0 <= result.accuracy <= 1.0
    assert result.duration_seconds > 0.0
    assert any(
        not torch.equal(actual, expected)
        for actual, expected in zip(model.parameters(), before, strict=True)
    )


def test_eval_one_epoch_matches_a_single_full_batch_loss() -> None:
    """Uneven batches must aggregate by sample, not by batch."""
    seed_everything(0)
    model = nn.Linear(4, 3)
    loss_fn = nn.CrossEntropyLoss()
    dataloader = build_dataloader(5, 2)

    features, targets = build_samples(5)
    with torch.no_grad():
        expected_loss = loss_fn(model(features), targets).item()

    result = eval_one_epoch(model, dataloader, loss_fn, DEVICE)

    assert result.sample_count == 5
    assert result.loss == pytest.approx(expected_loss, rel=1e-6)


def test_eval_one_epoch_leaves_parameters_and_gradients_untouched() -> None:
    seed_everything(0)
    model = nn.Linear(4, 3)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    eval_one_epoch(model, build_dataloader(5, 2), nn.CrossEntropyLoss(), DEVICE)

    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(model.parameters(), before, strict=True)
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_each_function_selects_the_matching_module_mode() -> None:
    seed_everything(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.Dropout(0.5))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.eval()
    train_one_epoch(model, build_dataloader(4, 2), optimizer, nn.CrossEntropyLoss(), DEVICE)
    assert model.training

    eval_one_epoch(model, build_dataloader(4, 2), nn.CrossEntropyLoss(), DEVICE)
    assert not model.training


def test_empty_dataloader_is_rejected() -> None:
    empty = DataLoader(TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)))

    with pytest.raises(ValueError, match="no samples"):
        eval_one_epoch(nn.Linear(4, 3), empty, nn.CrossEntropyLoss(), DEVICE)
