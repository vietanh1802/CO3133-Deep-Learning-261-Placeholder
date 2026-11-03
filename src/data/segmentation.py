"""Shared image-mask preprocessing and loaders for semantic segmentation."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.data.base import DataLoaders, ImageBatch
from src.utils.distributed import rank, world_size
from src.utils.seed import make_generator, seed_worker


class SegmentationDataset(Dataset[ImageBatch], ABC):
    """Load paired RGB images and integer masks using a dataset-specific decoder."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        *,
        image_size: tuple[int, int],
        augment: bool,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self.pairs = pairs
        self.stems = [image.stem for image, _ in pairs]
        self.image_size = image_size
        self.augment = augment
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    @abstractmethod
    def decode_target(self, mask: Image.Image) -> np.ndarray:
        """Return a two-dimensional integer label array for one mask image."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, mask_path = self.pairs[index]
        height, width = self.image_size
        with Image.open(image_path) as handle:
            image = handle.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        with Image.open(mask_path) as handle:
            labels = self.decode_target(handle)
        labels = Image.fromarray(labels).resize((width, height), Image.Resampling.NEAREST)

        pixels = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)
        target = torch.from_numpy(np.array(labels, dtype=np.int64))
        if self.augment and torch.rand(()) < 0.5:
            pixels, target = pixels.flip(-1), target.flip(-1)
        return (pixels - self.mean) / self.std, target

    def denormalize(self, images: Tensor) -> Tensor:
        """Map normalized images back to RGB in [0, 1]."""
        return (images * self.std.to(images) + self.mean.to(images)).clamp(0, 1)


def build_segmentation_loaders(
    train_dataset: Dataset[ImageBatch],
    validation_dataset: Dataset[ImageBatch],
    test_dataset: Dataset[ImageBatch],
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> DataLoaders[ImageBatch]:
    """Wrap segmentation splits, sharding only training across distributed ranks."""

    def loader(dataset: Dataset[ImageBatch], *, train: bool) -> DataLoader[ImageBatch]:
        sampler = (
            DistributedSampler(dataset, num_replicas=world_size(), rank=rank(), seed=seed)
            if train and world_size() > 1
            else None
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=make_generator(seed + rank()) if train else None,
        )

    return DataLoaders(
        train=loader(train_dataset, train=True),
        validation=loader(validation_dataset, train=False),
        test=loader(test_dataset, train=False),
    )
