"""Exercise the study CLI, partial failure/resume, and report-table safety."""

import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from scripts.evaluation import ablate_segmentation as study
from src.config.segmentation import load_segmentation_config
from src.data.suim import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[2]


def test_study_plans_runs_resumes_and_exports_paired_results(tmp_path, monkeypatch):
    base = load_segmentation_config(ROOT / "configs/segmentation/suim/ablation.yaml")
    base = replace(base, results_dir=tmp_path / "results", checkpoint_dir=tmp_path / "checkpoints")
    path = tmp_path / "reference.yaml"
    path.write_text(yaml.safe_dump(base.to_dict()))
    argv = ["study", "--config", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []

    def fake_training(command, *, check):
        assert check
        config = load_segmentation_config(command[command.index("--config") + 1])
        calls.append(config.name)
        run = config.results_dir / "a2" / config.name
        run.mkdir(parents=True, exist_ok=True)
        checkpoints = config.checkpoint_dir / config.name
        checkpoints.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": {"config": config.to_dict()}}, checkpoints / "last.pt")
        (checkpoints / "best.pt").write_bytes(b"checkpoint")
        if "resnet101" in config.name:
            assert ("--resume" in command) is not fail[0]
            if fail[0]:
                raise subprocess.CalledProcessError(1, command)
        score = 0.2 if "reference" in config.name else 0.3
        summary = {
            "config": config.to_dict(),
            "history": [{}] * config.epochs,
            "best_epoch": 1,
            "device": "cpu",
            "software": {},
            "world_size": 1,
            "effective_batch_size": 16,
            "git": {"commit": "test", "dirty": False},
            "parameters": 10,
            "trainable_parameters": 10,
            "training_seconds": 12.0,
            "peak_training_memory_mb": None,
            "validation_inference": {"milliseconds_per_image": 2.0},
            "validation": {
                "all": {
                    "mean_iou": score,
                    "mean_dice": score,
                    "per_class_iou": dict.fromkeys(CLASS_NAMES, score),
                },
                "boundary": {"mean_iou": score / 2},
            },
        }
        (run / "training_summary.json").write_text(json.dumps(summary))

    fail = [True]
    monkeypatch.setattr(study.subprocess, "run", fake_training)
    study.main()
    directory = base.results_dir / "a2" / "ablation"
    manifest_path = directory / "manifest.json"
    manifest = manifest_path.read_text()
    assert len(json.loads(manifest)) == 6
    assert len(list((directory / "configs").glob("*.yaml"))) == 6
    assert not calls
    # Planning is idempotent and never launches training.
    study.main()
    assert manifest_path.read_text() == manifest

    monkeypatch.setattr(sys, "argv", [*argv, "--run"])
    with pytest.raises(subprocess.CalledProcessError):
        study.main()
    assert len(calls) == 4
    assert not (directory / "comparisons.csv").exists()
    with pytest.raises(FileExistsError):
        study.main()
    assert len(calls) == 4

    fail[0] = False
    monkeypatch.setattr(sys, "argv", [*argv, "--run", "--resume"])
    study.main()
    # Three completed runs are reused, and the three remaining runs execute.
    assert len(calls) == 7
    with (directory / "comparisons.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert float(rows[0]["delta_miou_pp"]) == 0
    assert [float(row["delta_miou_pp"]) for row in rows[1:]] == pytest.approx([10] * 5)
    study.main()
    assert len(calls) == 7

    # A different training environment cannot silently enter the comparison table.
    variant = base.results_dir / "a2" / "suim_ablation_ce_dice_seed42" / "training_summary.json"
    record = json.loads(variant.read_text())
    record["device"] = "different GPU"
    variant.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="device differs"):
        study.main()

    # A changed plan is rejected without overwriting it or launching runs.
    changed = replace(base, learning_rate=0.02)
    path.write_text(yaml.safe_dump(changed.to_dict()))
    with pytest.raises(ValueError, match="study settings differ"):
        study.main()
    assert manifest_path.read_text() == manifest
    assert len(calls) == 7
