"""SUIM quality checks, class statistics, duplicate detection, and split analysis."""

import hashlib
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from src.data.suim import CLASS_NAMES, NUM_CLASSES, decode_mask


@dataclass(frozen=True)
class PairCheck:
    """Everything one pair contributes to the usable-sample count."""

    stem: str
    image_size: tuple[int, int] | None
    mask_size: tuple[int, int] | None
    off_code_pixels: int
    total_pixels: int
    class_pixels: tuple[int, ...]
    error: str | None = None

    @property
    def size_matches(self) -> bool:
        return self.image_size is not None and self.image_size == self.mask_size


def check_pair(image_path: Path, mask_path: Path) -> PairCheck:
    """Decode one image and its mask, reporting geometry and mask-code validity."""
    try:
        with Image.open(image_path) as handle:
            # Force full decoding to detect truncated images.
            handle.load()
            image_size = handle.size
        with Image.open(mask_path) as handle:
            mask = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            mask_size = handle.size
    except (OSError, ValueError) as exc:
        return PairCheck(
            stem=image_path.stem,
            image_size=None,
            mask_size=None,
            off_code_pixels=0,
            total_pixels=0,
            class_pixels=(),
            error=f"{type(exc).__name__}: {exc}",
        )

    # Require exact 0 or 255 channels for valid 3-bit codes.
    on_code = ((mask == 0) | (mask == 255)).all(axis=2)
    labels = decode_mask(mask)
    return PairCheck(
        stem=image_path.stem,
        image_size=image_size,
        mask_size=mask_size,
        off_code_pixels=int(on_code.size - on_code.sum()),
        total_pixels=int(on_code.size),
        class_pixels=tuple(
            int(count) for count in np.bincount(labels.ravel(), minlength=NUM_CLASSES)
        ),
    )


def usable_checks(checks: Iterable[PairCheck]) -> list[PairCheck]:
    """Select pairs with valid images, matching sizes, and exact mask colors.

    Notes:
        Recompressed masks can introduce false rare-class pixels at object edges.
        Only the eight exact 3-bit colors are accepted.
    """
    return [
        check
        for check in checks
        if check.error is None and check.size_matches and check.off_code_pixels == 0
    ]


@dataclass(frozen=True)
class ClassStat:
    """How much of the data one class owns, counted three ways."""

    index: int
    name: str
    image_count: int
    pixel_count: int
    pixel_share: float
    median_area_when_present: float


def class_summary(checks: Iterable[PairCheck]) -> list[ClassStat]:
    """Summarize class frequencies over usable pairs.

    Returns:
        Per-class image counts, pixel shares, and typical areas.
    """
    totals = np.zeros(NUM_CLASSES, dtype=np.int64)
    areas: list[list[float]] = [[] for _ in range(NUM_CLASSES)]
    for check in usable_checks(checks):
        counts = np.asarray(check.class_pixels, dtype=np.int64)
        totals += counts
        for index in np.flatnonzero(counts):
            areas[index].append(counts[index] / check.total_pixels)
    grand = int(totals.sum())
    return [
        ClassStat(
            index=index,
            name=CLASS_NAMES[index],
            image_count=len(areas[index]),
            pixel_count=int(totals[index]),
            pixel_share=int(totals[index]) / grand,
            median_area_when_present=float(np.median(areas[index])) if areas[index] else 0.0,
        )
        for index in range(NUM_CLASSES)
    ]


# A 9-by-8 grayscale thumbnail yields a 64-bit, resolution-tolerant difference hash.
DHASH_SIDE = 8


@dataclass(frozen=True)
class Fingerprint:
    """Two views of one image: its exact bytes and its coarse light-dark pattern."""

    stem: str
    split: str
    path: Path
    sha256: str
    dhash: int


def image_fingerprint(path: Path, split: str) -> Fingerprint:
    """Hash one image exactly and perceptually in a single read."""
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as handle:
        thumb = np.asarray(
            handle.convert("L").resize((DHASH_SIDE + 1, DHASH_SIDE), Image.Resampling.BILINEAR),
            dtype=np.int16,
        )
    bits = (thumb[:, 1:] > thumb[:, :-1]).ravel()
    return Fingerprint(
        stem=path.stem,
        split=split,
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        dhash=int(np.packbits(bits).view(">u8")[0]),
    )


def find_duplicate_candidates(
    fingerprints: Sequence[Fingerprint], max_distance: int
) -> list[tuple[Fingerprint, Fingerprint, int]]:
    """Find pairs whose hashes differ by at most max_distance bits.

    Notes:
        Hash matches are candidates; confirm duplicates against image pixels.
    """
    hashes = np.array([f.dhash for f in fingerprints], dtype=np.uint64)
    distance = np.bitwise_count(hashes[:, None] ^ hashes[None, :])
    rows, cols = np.nonzero(np.triu(distance <= max_distance, k=1))
    return [
        (fingerprints[i], fingerprints[j], int(distance[i, j]))
        for i, j in zip(rows, cols, strict=True)
    ]


def pixel_difference(path_a: Path, path_b: Path, side: int = 64) -> float:
    """Mean absolute RGB difference between two images at a common size, 0 to 255."""

    def load(path: Path) -> np.ndarray:
        with Image.open(path) as handle:
            return np.asarray(
                handle.convert("RGB").resize((side, side), Image.Resampling.BILINEAR),
                dtype=np.int16,
            )

    return float(np.abs(load(path_a) - load(path_b)).mean())


@dataclass(frozen=True)
class SplitAssignment:
    """One train/validation division of a split, with what it took to produce it."""

    train: list[str]
    val: list[str]
    seed: int
    n_splits: int
    group_count: int
    stratum_sizes: dict[str, int]


def grouped_stratified_split(
    checks: Sequence[PairCheck],
    duplicate_groups: Iterable[Iterable[str]],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> SplitAssignment:
    """Split groups without leaking duplicate photographs across splits.

    Notes:
        Stratification labels each image by its rarest class in this dataset.
        Fold 0 is validation; n_splits=5 gives a 20 percent validation split.
    """
    stems = [check.stem for check in checks]
    position = {stem: index for index, stem in enumerate(stems)}
    group = np.arange(len(stems))
    for members in duplicate_groups:
        inside = [position[stem] for stem in members if stem in position]
        for index in inside[1:]:
            group[index] = inside[0]

    present = np.array([[count > 0 for count in c.class_pixels] for c in checks])
    frequency = present.sum(axis=0)
    # Exclude absent classes so argmin selects a present class.
    stratum = np.where(present, frequency, frequency.max() + 1).argmin(axis=1)

    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_index, val_index = next(folds.split(stems, stratum, group))
    return SplitAssignment(
        train=sorted(stems[i] for i in train_index),
        val=sorted(stems[i] for i in val_index),
        seed=seed,
        n_splits=n_splits,
        group_count=len(np.unique(group)),
        stratum_sizes={
            CLASS_NAMES[index]: int((stratum == index).sum())
            for index in range(NUM_CLASSES)
            if (stratum == index).any()
        },
    )
