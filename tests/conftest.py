"""Small torchvision-shaped networks for download-free integration tests."""

from unittest.mock import Mock

import pytest
from torch import nn

import src.models.deeplabv3 as deeplabv3_module


class _TinyDeepLab(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.classifier = nn.Sequential(nn.Conv2d(256, 21, 1))
        self.aux_classifier = nn.Conv2d(256, 21, 1)

    def forward(self, images):
        features = self.backbone(images)
        return {"out": self.classifier(features)}


@pytest.fixture
def tiny_deeplab(monkeypatch):
    """Keep the wrapper/optimizer real while replacing only torchvision construction."""
    builder = Mock(side_effect=lambda **kwargs: _TinyDeepLab())
    monkeypatch.setattr(deeplabv3_module, "deeplabv3_resnet50", builder)
    return builder
