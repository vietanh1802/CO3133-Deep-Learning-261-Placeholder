"""Source modules and runnable experiment configurations remain loadable."""

import importlib
from pathlib import Path

import pytest

from src.config.classification import load_config
from src.config.segmentation import load_segmentation_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(
            path,
            marks=pytest.mark.skipif(
                not any(
                    line.strip() and not line.lstrip().startswith("#")
                    for line in path.read_text().splitlines()
                ),
                reason="Comment-only configuration placeholder",
            ),
            id=str(path.relative_to(ROOT)),
        )
        for path in sorted((ROOT / "configs").rglob("*.yaml"))
    ],
)
def test_experiment_config_loads(path):
    loader = load_segmentation_config if "segmentation" in path.parts else load_config
    config = loader(path)
    assert config.name
    assert config.batch_size > 0


def test_source_modules_import():
    for path in sorted((ROOT / "src").rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        importlib.import_module(".".join(parts))
