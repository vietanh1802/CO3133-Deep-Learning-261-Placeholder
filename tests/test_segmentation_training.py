import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.segmentation import load_segmentation_config
from src.data.suim import (
    SuimSegmentation,
    build_suim_loaders,
)
from src.data.transform import IMAGENET_MEAN, IMAGENET_STD
from src.evaluation.compute_cost import measure_inference
from src.models import UNet, build_model
from src.plots.segmentation import (
    plot_segmentation_curves,
)
from src.training.segmentation import fit_segmentation, warmup_cosine
from src.utils.environment import git_revision, software_versions
from src.utils.seed import make_generator, seed_everything

DEVICE = torch.device("cpu")
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


def test_unet_shape_and_proposal_size():
    model = build_model("unet", {"base_width": 32, "num_classes": 8})
    assert sum(p.numel() for p in model.parameters()) == pytest.approx(7.8e6, rel=0.01)
    assert model(torch.zeros(2, 3, 32, 48)).shape == (2, 8, 32, 48)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 3, 30, 48))


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


def test_baseline_config_loads(tmp_path):
    config = load_segmentation_config("configs/segmentation/suim/baseline.yaml")
    assert (config.model, config.batch_size * config.accumulation_steps) == ("unet", 16)
    assert (config.image_height, config.image_width, config.epochs) == (240, 320, 40)
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nmodel: unet\nval_fraction: 0.1\n")
    with pytest.raises(ValueError, match="unknown configuration keys: val_fraction"):
        load_segmentation_config(path)


def test_measure_inference_counts_every_image():
    images = torch.randn(5, 3, 16, 16)
    loader = DataLoader(TensorDataset(images, torch.zeros(5, 16, 16, dtype=torch.long)), 2)
    cost = measure_inference(UNet(num_classes=3, base_width=4), loader, DEVICE)
    assert (cost.images, cost.batch_size, cost.peak_memory_mb) == (5, 2, None)
    assert cost.milliseconds_per_image == pytest.approx(1000 * cost.seconds / 5)
    assert cost.images_per_second == pytest.approx(5 / cost.seconds)


def test_segmentation_curves_and_environment(tmp_path):
    path = tmp_path / "curves.png"
    plot_segmentation_curves([1.0, 0.8], [0.9, 0.85], [0.2, 0.3], path)
    assert path.stat().st_size > 0
    with pytest.raises(ValueError, match="match in length"):
        plot_segmentation_curves([1.0], [0.9], [0.2, 0.3], path)
    assert software_versions()["torch"] == torch.__version__
    assert set(git_revision()) == {"commit", "dirty"}
