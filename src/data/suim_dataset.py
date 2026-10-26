"""SUIM image/mask pairs prepared for segmentation training."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.base import DataLoaders, register_dataset
from src.data.suim import decode_mask, find_pairs
from src.utils.seed import make_generator, seed_worker

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_DIR = Path("results/a2/splits/suim")


class SuimSegmentation(Dataset[tuple[Tensor, Tensor]]):
    """Resized, normalized SUIM pairs for one of the train, val, or test splits."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        *,
        image_size: tuple[int, int] = (240, 320),
        augment: bool = False,
        split_dir: str | Path = SPLIT_DIR,
    ) -> None:
        data_dir = Path(data_dir)
        if split == "test":
            self.pairs = find_pairs(data_dir / "TEST").pairs
        elif split in ("train", "val"):
            pairs = find_pairs(data_dir / "train_val").pairs
            available = {image.stem: (image, mask) for image, mask in pairs}
            stems = (Path(split_dir) / f"{split}.txt").read_text(encoding="utf-8").split()
            missing = [stem for stem in stems if stem not in available]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} {split} pairs missing under {data_dir / 'train_val'}, "
                    f"e.g. {missing[0]!r}"
                )
            self.pairs = [available[stem] for stem in stems]
        else:
            raise ValueError("split must be train, val, or test")
        if not self.pairs:
            raise FileNotFoundError(f"no SUIM pairs found for split {split!r} under {data_dir}")

        self.stems = [image.stem for image, _ in self.pairs]
        self.image_size = image_size
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, mask_path = self.pairs[index]
        height, width = self.image_size
        with Image.open(image_path) as handle:
            image = handle.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        with Image.open(mask_path) as handle:
            labels = decode_mask(np.asarray(handle.convert("RGB"), dtype=np.uint8))
        labels = Image.fromarray(labels).resize((width, height), Image.Resampling.NEAREST)

        pixels = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)
        target = torch.from_numpy(np.array(labels, dtype=np.int64))
        if self.augment and torch.rand(()) < 0.5:
            pixels, target = pixels.flip(-1), target.flip(-1)
        return (pixels - self.mean) / self.std, target

    def denormalize(self, images: Tensor) -> Tensor:
        """Map normalized images back to RGB in [0, 1]."""
        return (images * self.std.to(images) + self.mean.to(images)).clamp(0, 1)


@register_dataset("suim")
def build_suim_loaders(
    *,
    data_dir: str | Path,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    image_height: int = 240,
    image_width: int = 320,
    split_dir: str | Path = SPLIT_DIR,
) -> DataLoaders[tuple[Tensor, Tensor]]:
    """Build train (flipped), validation, and test loaders from the committed split."""
    size = (image_height, image_width)

    def loader(split: str, *, train: bool) -> DataLoader[tuple[Tensor, Tensor]]:
        dataset = SuimSegmentation(
            data_dir, split, image_size=size, augment=train, split_dir=split_dir
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=make_generator(seed) if train else None,
        )

    return DataLoaders(
        train=loader("train", train=True),
        validation=loader("val", train=False),
        test=loader("test", train=False),
    )
