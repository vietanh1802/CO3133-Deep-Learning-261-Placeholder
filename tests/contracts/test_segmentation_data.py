"""Shared segmentation loading works with a mask encoding independent of SUIM."""

from typing import cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, TensorDataset

from src.data.base import ImageBatch
from src.data.segmentation import SegmentationDataset, build_segmentation_loaders


class _IndexedMasks(SegmentationDataset):
    def decode_target(self, mask: Image.Image) -> np.ndarray:
        return np.asarray(mask, dtype=np.uint8)


def test_shared_dataset_preserves_indexed_mask_labels_and_display_images(tmp_path):
    labels = np.tile([0, 0, 2, 2, 5, 5], (4, 1)).astype(np.uint8)
    pixels = np.repeat((labels * 20)[:, :, None], 3, axis=2)
    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    Image.fromarray(pixels).save(image_path)
    Image.fromarray(labels).save(mask_path)
    dataset = _IndexedMasks(
        [(image_path, mask_path)],
        image_size=(2, 3),
        augment=False,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    image, target = dataset[0]
    assert dataset.stems == ["image"]
    assert len(dataset) == 1
    assert image.shape == (3, 2, 3) and image.dtype == torch.float32
    assert target.dtype == torch.int64
    assert target.tolist() == [[0, 2, 5], [0, 2, 5]]
    displayed = dataset.denormalize(image)
    assert torch.all((displayed >= 0) & (displayed <= 1))
    torch.testing.assert_close(displayed, image * 0.5 + 0.5)


def test_shared_loaders_partition_training_but_keep_full_evaluation(monkeypatch):
    from src.data import segmentation

    images = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1)
    masks = torch.zeros(6, 2, 2, dtype=torch.long)
    train = cast(Dataset[ImageBatch], TensorDataset(images, masks))
    validation = cast(Dataset[ImageBatch], TensorDataset(images[:3], masks[:3]))
    test = cast(Dataset[ImageBatch], TensorDataset(images[:2], masks[:2]))
    monkeypatch.setattr(segmentation, "world_size", lambda: 2)
    shards = []
    for worker_rank in range(2):
        monkeypatch.setattr(segmentation, "rank", lambda value=worker_rank: value)
        loaders = build_segmentation_loaders(
            train_dataset=train,
            validation_dataset=validation,
            test_dataset=test,
            batch_size=2,
            seed=17,
            num_workers=0,
        )
        shards.append({int(value) for batch, _ in loaders.train for value in batch.flatten()})
        assert sum(len(target) for _, target in loaders.validation) == 3
        assert sum(len(target) for _, target in loaders.test) == 2
    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == set(range(6))
