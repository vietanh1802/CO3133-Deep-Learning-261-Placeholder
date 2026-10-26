import pytest

from src.config.segmentation import load_segmentation_config


def test_baseline_config_loads(tmp_path):
    config = load_segmentation_config("configs/segmentation/suim/baseline.yaml")
    assert (config.model, config.batch_size * config.accumulation_steps) == ("unet", 16)
    assert (config.image_height, config.image_width, config.epochs) == (240, 320, 40)
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nmodel: unet\nval_fraction: 0.1\n")
    with pytest.raises(ValueError, match="unknown configuration keys: val_fraction"):
        load_segmentation_config(path)
