#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
target=${1:-"$root/data/suim"}

if [[ -f "$target/.download-complete" ]]; then
    for directory in train_val/images train_val/masks TEST/images TEST/masks; do
        [[ -d "$target/$directory" ]] || { echo "Missing $target/$directory" >&2; exit 1; }
    done
    echo "SUIM already downloaded: $target"
    exit 0
fi
if [[ -e "$target" ]]; then
    echo "Refusing to overwrite existing dataset: $target" >&2
    exit 1
fi

mkdir -p "$(dirname "$target")"
staging=$(mktemp -d "$(dirname "$target")/.suim-download.XXXXXX")
trap 'rm -rf "$staging"' EXIT

# Official archive linked at https://irvlab.cs.umn.edu/resources/suim-dataset.
uvx --from gdown==5.2.0 gdown 1uEnlqKrlt6lITc_i80NTtb7iHGcO47sU --output "$staging/SUIM.zip"
python3 - "$staging" "$target" <<'PY'
import pathlib
import shutil
import sys
import zipfile

staging, target = map(pathlib.Path, sys.argv[1:])
extracted = staging / "extracted"
with zipfile.ZipFile(staging / "SUIM.zip") as archive:
    members = [
        item for item in archive.infolist()
        if not (item.filename == "/" and item.is_dir() and item.file_size == 0)
    ]
    for item in members:
        if not (extracted / item.filename).resolve().is_relative_to(extracted.resolve()):
            raise ValueError(f"Unsafe archive path: {item.filename}")
    archive.extractall(extracted, members=members)
roots = [path.parent for path in extracted.rglob("train_val") if path.is_dir()]
roots = [path for path in roots if (path / "TEST").is_dir()]
if len(roots) != 1:
    raise ValueError("Expected one SUIM root containing train_val and TEST")
source = roots[0]
for directory in ("train_val/images", "train_val/masks", "TEST/images", "TEST/masks"):
    path = source / directory
    if not path.is_dir() or not any(item.is_file() for item in path.iterdir()):
        raise ValueError(f"Missing or empty dataset directory: {directory}")
(source / ".download-complete").touch()
shutil.move(str(source), str(target))
print(f"SUIM downloaded: {target}")
PY
