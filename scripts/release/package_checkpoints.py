"""Package best checkpoints for one dataset as an optional release asset."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import torch

DATASETS = ("fashion_mnist", "mnist", "cifar10")


def parse_args() -> argparse.Namespace:
    """Parse checkpoint packaging arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, choices=DATASETS)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def package_checkpoints(data: str, checkpoint_dir: Path, output_dir: Path) -> Path:
    """Write one dataset-pure checkpoint archive and its manifest."""
    selected: list[tuple[Path, dict[str, Any]]] = []
    for checkpoint_path in sorted(checkpoint_dir.glob("*/best.pt")):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"checkpoint has invalid metadata: {checkpoint_path}")
        config = metadata.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"checkpoint has invalid config metadata: {checkpoint_path}")
        if config.get("dataset") == data:
            selected.append((checkpoint_path, metadata))

    if not selected:
        raise FileNotFoundError(f"no best checkpoints found for dataset {data!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{data}-checkpoints.zip"
    manifest = {
        "dataset": data,
        "checkpoints": [
            {
                "run": checkpoint_path.parent.name,
                "epoch": metadata["epoch"],
                "best_validation_loss": metadata["best_validation_loss"],
                "bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256(checkpoint_path),
            }
            for checkpoint_path, metadata in selected
        ],
    }

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for checkpoint_path, _ in selected:
            archive.write(
                checkpoint_path,
                arcname=f"checkpoints/{checkpoint_path.parent.name}/best.pt",
            )
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")

    return archive_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Build one optional checkpoint release asset."""
    args = parse_args()
    archive_path = package_checkpoints(args.data, args.checkpoint_dir, args.output_dir)
    print(archive_path)


if __name__ == "__main__":
    main()
