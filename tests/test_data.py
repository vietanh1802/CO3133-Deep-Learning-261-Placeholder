from typing import Any, cast

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import build_dataloaders, class_names, register_dataset
from src.data.base import DataLoaders, ImageClassificationBatch


def test_dataset_factory_dispatches_to_named_builder() -> None:
    dataset = TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long))
    loader = DataLoader(dataset)
    expected = cast(
        DataLoaders[ImageClassificationBatch],
        DataLoaders(train=loader, validation=loader, test=loader),
    )
    received: list[dict[str, Any]] = []

    @register_dataset("test_dataset")
    def builder(**config: Any) -> DataLoaders[ImageClassificationBatch]:
        received.append(config)
        return expected

    actual = build_dataloaders("test-dataset", {"data_dir": "data", "batch_size": 64})
    assert actual is expected
    assert received == [{"data_dir": "data", "batch_size": 64}]


def test_dataset_factory_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_dataloaders("unknown", {})


@pytest.mark.parametrize("name", ["fashion_mnist", "mnist", "cifar10"])
def test_builtin_datasets_are_registered(name: str) -> None:
    with pytest.raises(TypeError, match="data_dir"):
        build_dataloaders(name, {})


def test_builtin_dataset_class_names_match_label_order() -> None:
    assert class_names("mnist") == tuple(str(index) for index in range(10))
    assert class_names("fashion-mnist")[0] == "T-shirt/top"
    assert class_names("cifar10")[-1] == "truck"
