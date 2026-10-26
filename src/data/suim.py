"""SUIM file pairing, mask decoding, and segmentation data loading."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.base import DataLoaders, ImageBatch, register_dataset
from src.data.segmentation import SegmentationDataset, build_segmentation_loaders
from src.data.transform import IMAGENET_MEAN, IMAGENET_STD

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
MASK_SUFFIX = ".bmp"


@dataclass(frozen=True)
class SplitPairs:
    """One split's matched image/mask pairs plus the files left unmatched."""

    name: str
    pairs: list[tuple[Path, Path]]
    images_without_mask: list[str]
    masks_without_image: list[str]


def find_pairs(split_dir: str | Path) -> SplitPairs:
    """Match images to masks by filename stem inside one SUIM split."""
    split_dir = Path(split_dir)
    images = {
        path.stem: path
        for path in (split_dir / "images").glob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    masks = {
        path.stem: path
        for path in (split_dir / "masks").glob("*")
        if path.suffix.lower() == MASK_SUFFIX
    }
    matched = sorted(images.keys() & masks.keys())
    return SplitPairs(
        name=split_dir.name,
        pairs=[(images[stem], masks[stem]) for stem in matched],
        images_without_mask=sorted(images.keys() - masks.keys()),
        masks_without_image=sorted(masks.keys() - images.keys()),
    )


# The authors encode each class as a 3-bit RGB colour, so a channel is meant to be
# exactly 0 or exactly 255 and the bit pattern read as binary is the class index.
MASK_BIT_THRESHOLD = 128
NUM_CLASSES = 8
CLASS_NAMES = ("BW", "HD", "PF", "WR", "RO", "RI", "FV", "SR")


def decode_mask(rgb: np.ndarray) -> np.ndarray:
    """Turn an ``(H, W, 3)`` uint8 SUIM mask into an ``(H, W)`` array of class indices."""
    bits = (rgb >= MASK_BIT_THRESHOLD).astype(np.uint8)
    return bits[:, :, 0] * 4 + bits[:, :, 1] * 2 + bits[:, :, 2]


SPLIT_DIR = Path("results/a2/splits/suim")


class SuimSegmentation(SegmentationDataset):
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
            pairs = find_pairs(data_dir / "TEST").pairs
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
            pairs = [available[stem] for stem in stems]
        else:
            raise ValueError("split must be train, val, or test")
        if not pairs:
            raise FileNotFoundError(f"no SUIM pairs found for split {split!r} under {data_dir}")

        super().__init__(
            pairs,
            image_size=image_size,
            augment=augment,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

    def decode_target(self, mask: Image.Image) -> np.ndarray:
        """Decode SUIM's three-bit RGB class colours."""
        return decode_mask(np.asarray(mask.convert("RGB"), dtype=np.uint8))


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
) -> DataLoaders[ImageBatch]:
    """Build train (flipped), validation, and test loaders from the committed split."""
    size = (image_height, image_width)

    return build_segmentation_loaders(
        SuimSegmentation(data_dir, "train", image_size=size, augment=True, split_dir=split_dir),
        SuimSegmentation(data_dir, "val", image_size=size, split_dir=split_dir),
        SuimSegmentation(data_dir, "test", image_size=size, split_dir=split_dir),
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
    )
