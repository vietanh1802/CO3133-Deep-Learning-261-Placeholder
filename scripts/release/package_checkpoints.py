"""Package A1 checkpoints into checkpoints/a1.zip, optionally filtering by dataset."""

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import torch

DATASETS = ("fashion_mnist", "mnist", "cifar10")


def parse_args() -> argparse.Namespace:
    """Parse checkpoint packaging arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=DATASETS, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def package_checkpoints(data: str | None, checkpoint_dir: Path, output_dir: Path) -> Path:
    """Package a downloaded checkpoint folder or an existing assignment archive."""
    sources: dict[str, bytes] = {}
    existing = checkpoint_dir / "a1.zip"
    if existing.exists():
        with ZipFile(existing) as archive:
            for member in archive.namelist():
                if member.endswith("/best.pt"):
                    name = Path(member).parent.name
                    if name in sources:
                        raise ValueError(f"duplicate run name in archive: {name}")
                    sources[name] = archive.read(member)
    for path in sorted(checkpoint_dir.glob("*/best.pt")):
        sources[path.parent.name] = path.read_bytes()
    selected: list[tuple[str, bytes, dict[str, Any]]] = []
    for run_name, checkpoint_bytes in sorted(sources.items()):
        payload = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("config"), dict):
            raise ValueError(f"checkpoint has invalid metadata: {run_name}")
        if data is None or metadata["config"].get("dataset") == data:
            selected.append((run_name, checkpoint_bytes, metadata))
    if not selected:
        raise FileNotFoundError(f"no best checkpoints found for dataset {data!r}")
    if len({name for name, _, _ in selected}) != len(selected):
        raise ValueError("duplicate run names in checkpoint source")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / (f"{data}-checkpoints.zip" if data else "a1.zip")
    temporary_path = archive_path.with_suffix(".zip.tmp")
    manifest = {
        "assignment": "a1",
        "dataset": data,
        "checkpoints": [
            {
                "run": name,
                "epoch": metadata["epoch"],
                "best_validation_loss": metadata["best_validation_loss"],
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content, metadata in selected
        ],
    }
    try:
        with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED) as archive:
            for name, content, _ in selected:
                archive.writestr(f"checkpoints/{name}/best.pt", content)
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        temporary_path.replace(archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return archive_path


def main() -> None:
    """Build one optional checkpoint release asset."""
    args = parse_args()
    print(package_checkpoints(args.data, args.checkpoint_dir, args.output_dir))


if __name__ == "__main__":
    main()
