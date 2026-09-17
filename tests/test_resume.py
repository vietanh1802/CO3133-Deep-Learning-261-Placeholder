import random

import numpy as np
import torch
from torch import nn

from src.training.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from src.utils.seed import seed_everything


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
