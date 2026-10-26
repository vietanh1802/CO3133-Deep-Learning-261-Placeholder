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


def test_package_all_a1_checkpoints_uses_assignment_archive_name(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_checkpoint(source / "fashion_mnist_linear_seed42/best.pt", "fashion_mnist")
    _write_checkpoint(source / "mnist_linear_seed42/best.pt", "mnist")
    archive_path = package_checkpoints(None, source, tmp_path / "checkpoints")
    assert archive_path.name == "a1.zip"
    with ZipFile(archive_path) as archive:
        assert "checkpoints/fashion_mnist_linear_seed42/best.pt" in archive.namelist()
        assert "checkpoints/mnist_linear_seed42/best.pt" in archive.namelist()


def test_repackage_preserves_existing_runs_when_loose_checkpoint_is_added(tmp_path: Path) -> None:
    source = tmp_path / "checkpoints"
    _write_checkpoint(source / "first/best.pt", "fashion_mnist")
    archive = package_checkpoints(None, source, source)
    (source / "first/best.pt").unlink()
    _write_checkpoint(source / "second/best.pt", "mnist")
    assert package_checkpoints(None, source, source) == archive
    with ZipFile(archive) as stored:
        assert "checkpoints/first/best.pt" in stored.namelist()
        assert "checkpoints/second/best.pt" in stored.namelist()
