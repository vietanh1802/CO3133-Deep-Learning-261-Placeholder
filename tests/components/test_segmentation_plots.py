import pytest
import torch

from src.plots.segmentation import (
    plot_segmentation_curves,
)
from src.utils.environment import git_revision, software_versions


def test_segmentation_curves_and_environment(tmp_path):
    path = tmp_path / "curves.png"
    plot_segmentation_curves([1.0, 0.8], [0.9, 0.85], [0.2, 0.3], path)
    assert path.stat().st_size > 0
    with pytest.raises(ValueError, match="match in length"):
        plot_segmentation_curves([1.0], [0.9], [0.2, 0.3], path)
    assert software_versions()["torch"] == torch.__version__
    assert set(git_revision()) == {"commit", "dirty"}
