import numpy as np
import pytest
import torch
from PIL import Image

from src.data.suim import (
    SuimSegmentation,
    build_suim_loaders,
)
from src.data.transform import IMAGENET_MEAN, IMAGENET_STD

COLOURS = {0: (0, 0, 0), 1: (0, 0, 255), 4: (255, 0, 0), 7: (255, 255, 255)}


def _write_pair(directory, stem, labels):
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "masks").mkdir(parents=True, exist_ok=True)
    mask = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label, colour in COLOURS.items():
        mask[labels == label] = colour
    image = np.repeat((labels * 30).astype(np.uint8)[:, :, None], 3, axis=2)
    Image.fromarray(image).save(directory / "images" / f"{stem}.png")
    Image.fromarray(mask).save(directory / "masks" / f"{stem}.bmp")


@pytest.fixture
def suim_tree(tmp_path):
    labels = np.zeros((32, 48), dtype=np.int64)
    labels[:, :12] = 1
    labels[:16, 24:] = 4
    labels[16:, 36:] = 7
    for stem in ("a", "b", "c"):
        _write_pair(tmp_path / "suim" / "train_val", stem, labels)
    _write_pair(tmp_path / "suim" / "TEST", "t", labels)
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.txt").write_text("a\nb\n")
    (splits / "val.txt").write_text("c\n")
    return tmp_path / "suim", splits, labels


def test_dataset_decodes_and_resizes(suim_tree):
    data_dir, splits, labels = suim_tree
    dataset = SuimSegmentation(data_dir, "train", image_size=(16, 24), split_dir=splits)
    image, target = dataset[0]
    assert dataset.stems == ["a", "b"]
    assert image.shape == (3, 16, 24) and image.dtype == torch.float32
    assert target.shape == (16, 24) and target.dtype == torch.int64
    assert set(target.unique().tolist()) == {0, 1, 4, 7}
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    assert torch.allclose(dataset.denormalize(image), image * std + mean)


def test_flip_moves_image_and_mask_together(suim_tree):
    data_dir, splits, labels = suim_tree
    dataset = SuimSegmentation(
        data_dir, "train", image_size=(32, 48), augment=True, split_dir=splits
    )
    unflipped = torch.from_numpy(labels)
    torch.manual_seed(0)
    flips = set()
    for _ in range(20):
        image, target = dataset[0]
        assert torch.equal(target * 30, (dataset.denormalize(image)[0] * 255).round().long())
        flips.add(torch.equal(target, unflipped.flip(-1)))
    assert flips == {True, False}


def test_loaders_follow_split_files(suim_tree):
    data_dir, splits, _ = suim_tree
    loaders = build_suim_loaders(
        data_dir=data_dir, batch_size=1, seed=0, image_height=16, image_width=32, split_dir=splits
    )
    sizes = [len(loader) for loader in (loaders.train, loaders.validation, loaders.test)]
    assert sizes == [2, 1, 1]
    (splits / "val.txt").write_text("c\nmissing\n")
    with pytest.raises(FileNotFoundError, match="missing"):
        SuimSegmentation(data_dir, "val", split_dir=splits)
