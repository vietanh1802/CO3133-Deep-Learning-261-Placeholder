import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.compute_cost import measure_inference
from src.models import UNet

DEVICE = torch.device("cpu")


def test_measure_inference_counts_every_image():
    images = torch.randn(5, 3, 16, 16)
    loader = DataLoader(TensorDataset(images, torch.zeros(5, 16, 16, dtype=torch.long)), 2)
    cost = measure_inference(UNet(num_classes=3, base_width=4), loader, DEVICE)
    assert (cost.images, cost.batch_size, cost.peak_memory_mb) == (5, 2, None)
    assert cost.milliseconds_per_image == pytest.approx(1000 * cost.seconds / 5)
    assert cost.images_per_second == pytest.approx(5 / cost.seconds)
