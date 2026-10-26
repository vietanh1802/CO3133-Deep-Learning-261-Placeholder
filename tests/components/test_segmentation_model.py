import pytest
import torch

from src.models import build_model

DEVICE = torch.device("cpu")


def test_unet_shape_and_proposal_size():
    model = build_model("unet", {"base_width": 32, "num_classes": 8})
    assert sum(p.numel() for p in model.parameters()) == pytest.approx(7.8e6, rel=0.01)
    assert model(torch.zeros(2, 3, 32, 48)).shape == (2, 8, 32, 48)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 3, 30, 48))
