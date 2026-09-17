import random

import numpy as np
import pytest
import torch
from loguru import logger

from src.utils.config import ExperimentConfig, load_config
from src.utils.logging import progress_bar, setup_logging
from src.utils.seed import capture_rng_state, restore_rng_state, seed_everything


def test_load_config_and_serialize_paths(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "name: baseline\nmodel: linear\nbatch_size: 32\ndata_dir: local-data\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config == ExperimentConfig(
        name="baseline",
        model="linear",
        batch_size=32,
        data_dir=type(config.data_dir)("local-data"),
    )
    assert config.to_dict()["data_dir"] == "local-data"


def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("name: baseline\nmodel: linear\ntyop: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown configuration keys: tyop"):
        load_config(path)


def test_seed_and_rng_state_round_trip():
    seed_everything(7)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(1))

    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(1))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_setup_logging_writes_file(tmp_path):
    path = tmp_path / "logs" / "run.log"
    setup_logging(log_file=path)

    logger.info("utility logging works")

    assert "utility logging works" in path.read_text(encoding="utf-8")


def test_progress_bar_preserves_iterable_values():
    assert list(progress_bar(range(3), disable=True)) == [0, 1, 2]
