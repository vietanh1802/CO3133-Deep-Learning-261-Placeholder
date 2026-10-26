import json
from pathlib import Path
from zipfile import ZipFile

import torch

from scripts.release.package_checkpoints import package_checkpoints


def _write_checkpoint(path: Path, dataset: str) -> None:
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "metadata": {
                "epoch": 4,
                "global_step": 12,
                "best_validation_loss": 0.25,
                "config": {"dataset": dataset},
            }
        },
        path,
    )


def test_package_checkpoints_includes_only_requested_dataset(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    _write_checkpoint(checkpoint_dir / "fashion_mnist_linear_seed42" / "best.pt", "fashion_mnist")
    _write_checkpoint(checkpoint_dir / "mnist_linear_seed42" / "best.pt", "mnist")

    archive_path = package_checkpoints("fashion_mnist", checkpoint_dir, tmp_path / "artifacts")

    assert archive_path.name == "fashion_mnist-checkpoints.zip"
    with ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == [
            "checkpoints/fashion_mnist_linear_seed42/best.pt",
            "manifest.json",
        ]
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["dataset"] == "fashion_mnist"
    assert manifest["checkpoints"][0]["run"] == "fashion_mnist_linear_seed42"
