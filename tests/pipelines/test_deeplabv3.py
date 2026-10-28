"""Exercise DeepLabV3 CLI integration with real synthetic SUIM files and a tiny backbone."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from scripts.evaluation import eval_segmentation
from scripts.training import train_segmentation
from src.config.segmentation import load_segmentation_config
from src.data.suim import SuimSegmentation, build_suim_loaders
from src.training import segmentation


def test_deeplab_cli_resumes_two_rates_and_exports_validation(tmp_path, monkeypatch, tiny_deeplab):
    data = tmp_path / "data"
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.txt").write_text("train0\ntrain1\ntrain2\n")
    (splits / "val.txt").write_text("val0\nval1\n")
    rng = np.random.default_rng(7)
    labels = (np.indices((16, 24)).sum(0) % 8).astype(np.uint8)
    palette = np.array([[255 * ((i >> bit) & 1) for bit in (2, 1, 0)] for i in range(8)])
    for stem in ("train0", "train1", "train2", "val0", "val1", "test0"):
        folder = data / ("TEST" if stem.startswith("test") else "train_val")
        for child in ("images", "masks"):
            (folder / child).mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)).save(
            folder / "images" / f"{stem}.png"
        )
        Image.fromarray(palette[labels].astype(np.uint8)).save(folder / "masks" / f"{stem}.bmp")

    original_getitem = SuimSegmentation.__getitem__

    def validation_only(self, index):
        assert not self.stems[index].startswith("test"), "training must not inspect test samples"
        return original_getitem(self, index)

    monkeypatch.setattr(SuimSegmentation, "__getitem__", validation_only)
    monkeypatch.setattr(train_segmentation, "initialize", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        train_segmentation,
        "build_dataloaders",
        lambda dataset, options: build_suim_loaders(**options, split_dir=splits),
    )
    root = Path(__file__).resolve().parents[2]
    original = load_segmentation_config(root / "configs/segmentation/suim/pretrained.yaml")
    config = replace(
        original,
        name="deeplab_smoke",
        model_config={**original.model_config, "pretrained": False},
        batch_size=2,
        accumulation_steps=2,
        epochs=2,
        warmup_epochs=0,
        image_height=16,
        image_width=24,
        num_workers=0,
        data_dir=data,
        results_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    argv = ["train_segmentation", "--config", str(config_path)]
    monkeypatch.setattr(sys, "argv", argv)

    class InterruptedRunError(Exception):
        pass

    original_save = segmentation.save_training_state

    def save_then_interrupt(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        if Path(path).name == "last.pt":
            raise InterruptedRunError

    monkeypatch.setattr(segmentation, "save_training_state", save_then_interrupt)
    with pytest.raises(InterruptedRunError):
        train_segmentation.run()
    monkeypatch.setattr(segmentation, "save_training_state", original_save)
    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    train_segmentation.run()

    run = config.results_dir / "a2" / config.name
    summary = json.loads((run / "training_summary.json").read_text())
    assert [record["epoch"] for record in summary["history"]] == [1, 2]
    assert summary["config"]["backbone_learning_rate"] == 1e-4
    assert summary["effective_batch_size"] == 4
    assert summary["checkpoint_selection"] == "highest validation mIoU"
    assert 0 <= summary["validation"]["all"]["mean_iou"] <= 1
    checkpoint = torch.load(config.checkpoint_dir / config.name / "last.pt", weights_only=True)
    groups = checkpoint["optimizer"]["param_groups"]
    assert [group["name"] for group in groups] == ["head", "backbone"]
    assert checkpoint["scheduler"]["base_lrs"] == pytest.approx([1e-3, 1e-4])
    evidence = torch.load(run / "val_evidence.pt", weights_only=True)
    assert evidence["split"] == "validation"
    assert evidence["sample_ids"] == ["val0", "val1"]
    assert evidence["images"].shape == (2, 3, 16, 24)
    assert evidence["targets"].shape == evidence["predictions"].shape == (2, 16, 24)
    assert torch.isfinite(evidence["images"]).all()
    assert evidence["images"].min() >= 0 and evidence["images"].max() <= 1

    output = run / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_segmentation", "--evidence", str(run / "val_evidence.pt"), "--output", str(output)],
    )
    eval_segmentation.main()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["metrics"]["all"]["pixel_count"] == 2 * 16 * 24
    for path in (
        config.checkpoint_dir / config.name / "best.pt",
        run / "learning_curves.png",
        run / "learning_curves.pdf",
        output / "confusion_matrix.png",
        output / "examples.png",
    ):
        assert path.stat().st_size > 0
