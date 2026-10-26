"""Move legacy A1 artifacts into outputs/a1_v1 without overwriting different files."""

import argparse
import hashlib
import shutil
from pathlib import Path


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def migrate_outputs(repo_dir: Path) -> int:
    """Preflight all collisions, then move artifacts; repeated calls are safe."""
    repo_dir = repo_dir.resolve()
    output_root = repo_dir / "outputs/a1_v1"
    sources = [
        (repo_dir / "checkpoints", output_root / "checkpoints"),
        (repo_dir / "results/a1", output_root / "results"),
        (output_root / "results", output_root / "results"),
    ]
    moves: dict[Path, Path] = {}
    targets: dict[Path, Path] = {}
    for source_root, destination_root in sources:
        if not source_root.exists():
            continue
        for source in sorted(source_root.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            if destination_root.name == "results" and source.suffix == ".log":
                destination = output_root / "console" / relative
            if source == destination:
                continue
            for path in (source, destination):
                if not path.resolve().is_relative_to(repo_dir):
                    raise ValueError(f"artifact path leaves repository: {path}")
            previous = targets.get(destination)
            if previous is not None and _digest(previous) != _digest(source):
                raise FileExistsError(f"conflicting source files for {destination}")
            if destination.exists() and (
                not destination.is_file() or _digest(source) != _digest(destination)
            ):
                raise FileExistsError(f"refusing to overwrite different artifact: {destination}")
            moves[source] = destination
            targets[destination] = source

    for source, destination in moves.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            source.unlink()  # An identical copy was checked above.
        else:
            shutil.move(str(source), str(destination))

    for source_root, _ in sources[:2]:
        if source_root.exists():
            directories = [p for p in source_root.rglob("*") if p.is_dir()]
            for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
                if not any(directory.iterdir()):
                    directory.rmdir()
            if not any(source_root.iterdir()):
                source_root.rmdir()
    legacy_results = repo_dir / "results"
    if legacy_results.exists() and not any(legacy_results.iterdir()):
        legacy_results.rmdir()
    for name in ("checkpoints", "results", "configs", "console"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    return len(moves)


def main() -> None:
    """Migrate artifacts for an existing checkout after pulling the new layout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    count = migrate_outputs(args.repo_dir)
    print(f"Moved {count} files into {args.repo_dir.resolve() / 'outputs/a1_v1'}")


if __name__ == "__main__":
    main()
