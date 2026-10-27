from pathlib import Path

import pytest

from src.config.segmentation import load_segmentation_config


@pytest.mark.parametrize(
    ("batch_size", "accumulation_steps", "height", "width", "epochs"),
    [(4, 4, 240, 320, 40), (8, 2, 32, 48, 2)],
)
def test_segmentation_config_preserves_experiment_settings(
    tmp_path, batch_size, accumulation_steps, height, width, epochs
):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        f"name: toy\nmodel: unet\nbatch_size: {batch_size}\n"
        f"accumulation_steps: {accumulation_steps}\nimage_height: {height}\n"
        f"image_width: {width}\nepochs: {epochs}\nwarmup_epochs: 0\n"
    )
    config = load_segmentation_config(path)
    assert (config.batch_size, config.accumulation_steps) == (batch_size, accumulation_steps)
    assert (config.image_height, config.image_width, config.epochs) == (height, width, epochs)


def test_segmentation_config_rejects_unknown_settings(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nmodel: unet\nval_fraction: 0.1\n")
    with pytest.raises(ValueError, match="unknown configuration keys: val_fraction"):
        load_segmentation_config(path)


def test_committed_pretrained_config_preserves_fine_tuning_settings():
    root = Path(__file__).resolve().parents[2]
    config = load_segmentation_config(root / "configs/segmentation/suim/pretrained.yaml")
    assert config.model == "deeplabv3"
    assert config.model_config == {
        "num_classes": 8,
        "pretrained": True,
        "freeze_batch_norm": True,
    }
    assert config.learning_rate == 1e-3
    assert config.backbone_learning_rate == 1e-4
    assert config.to_dict()["backbone_learning_rate"] == 1e-4
    assert (config.image_height, config.image_width, config.seed) == (240, 320, 42)
    assert config.batch_size * config.accumulation_steps == 16


@pytest.mark.parametrize("rate", ["0", "-0.001", ".nan", ".inf"])
def test_config_rejects_invalid_backbone_learning_rate(tmp_path, rate):
    path = tmp_path / "invalid.yaml"
    path.write_text(f"name: x\nmodel: deeplabv3\nbackbone_learning_rate: {rate}\n")
    with pytest.raises(ValueError, match="backbone_learning_rate must be finite and positive"):
        load_segmentation_config(path)
