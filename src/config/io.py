"""Shared YAML validation and path conversion for experiment configurations."""

from dataclasses import MISSING, Field
from pathlib import Path
from typing import Any

import yaml

PATH_KEYS = ("data_dir", "checkpoint_dir", "results_dir")


def load_config_values(path: str | Path, config_fields: tuple[Field[Any], ...]) -> dict[str, Any]:
    """Read known dataclass fields and convert configured directories to paths."""
    with Path(path).open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    unknown_keys = set(raw) - {item.name for item in config_fields}
    if unknown_keys:
        raise ValueError(f"unknown configuration keys: {', '.join(sorted(unknown_keys))}")
    required_keys = {
        item.name
        for item in config_fields
        if item.default is MISSING and item.default_factory is MISSING
    }
    missing_keys = required_keys - set(raw)
    if missing_keys:
        raise ValueError(f"missing required configuration keys: {', '.join(sorted(missing_keys))}")

    for key in PATH_KEYS:
        if key in raw:
            raw[key] = Path(raw[key])
    return raw


def serialize_paths(values: dict[str, Any]) -> dict[str, Any]:
    """Convert a fresh dataclass dictionary's directory paths to strings."""
    for key in PATH_KEYS:
        values[key] = str(values[key])
    return values
