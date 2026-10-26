"""Shared updates preserve accumulation and legacy checkpoint contracts."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from src.training.epoch import run_training_epoch
from src.training.segmentation import (
    SegmentationCheckpointMetadata,
    load_training_state,
    save_training_state,
)


class _MappingModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, images):
        return {"out": self.linear(images)}


def test_accumulation_matches_full_batches_including_partial_group():
    torch.manual_seed(19)
    images = torch.randn(5, 4)
    targets = torch.randint(0, 3, (5,))
    model = _MappingModel(4, 3)
    expected = nn.Linear(4, 3)
    expected.load_state_dict(model.linear.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, 1, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss()
    result = run_training_epoch(
        model,
        DataLoader(TensorDataset(images, targets), batch_size=2),
        optimizer,
        loss_fn,
        torch.device("cpu"),
        scheduler=scheduler,
        accumulation_steps=2,
    )
    for start, end in ((0, 4), (4, 5)):
        reference_optimizer.zero_grad(set_to_none=True)
        loss_fn(expected(images[start:end]), targets[start:end]).backward()
        reference_optimizer.step()
        reference_scheduler.step()
    for actual, reference in zip(model.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(actual, reference)
    assert result.sample_count == 5
    assert scheduler.last_epoch == reference_scheduler.last_epoch == 2
    assert optimizer.param_groups[0]["lr"] == reference_optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize("segmentation", [False, True])
def test_legacy_checkpoint_without_distributed_fields_still_loads(tmp_path, segmentation):
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    path = tmp_path / "legacy.pt"
    if segmentation:
        metadata = SegmentationCheckpointMetadata(2, 4, 0.3, {"name": "legacy"})
        save_training_state(path, model, optimizer, scheduler, metadata)
    else:
        metadata = CheckpointMetadata(2, 4, 0.3, {"name": "legacy"})
        save_checkpoint(path, model, optimizer, metadata)
    payload = torch.load(path, weights_only=True)
    payload.pop("rng_states")
    payload.pop("world_size")
    torch.save(payload, path)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        model.weight.zero_()
    if segmentation:
        restored = load_training_state(path, model, optimizer, scheduler)
    else:
        restored = load_checkpoint(path, model, optimizer)
    assert restored == metadata
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
