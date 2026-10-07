"""File-level inventory of the SUIM underwater segmentation dataset.

This module answers how many samples the dataset really has. A segmentation sample
is an image *and* its mask, so the count is the number of filename stems present in
both folders, and a pair is only usable once both files decode and agree on size.
"""

import hashlib
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

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
            # load() forces the full decode, which is what catches a truncated JPEG;
            # reading only the header would report a size for a half-written file.
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

    # Exact corner test, not a tolerance band: a pixel is validly coded only when
    # every channel is 0 or 255. Anything else means the 3-bit assumption is unsafe.
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
    """Keep the pairs this project trains and evaluates on.

    A pair is usable when it decodes, when image and mask agree on size, and when
    every mask pixel sits exactly on one of the eight 3-bit colours. The last rule
    is not pedantry: in a recompressed mask the blue channel of RI magenta dips
    below the threshold along object edges, and the fringe decodes as RO. Measured
    over train_val, RO pixels touch an RI pixel in 0 percent of a clean mask but in
    97 percent of a recompressed one, so those masks carry phantom rare-class
    regions. The definition lives here once rather than in each caller.
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
    """Per-class image counts, pixel shares and typical area over the usable pairs.

    Pixel share and image count can disagree: a class in many images that covers a
    sliver of each is common by one count and rare by the other, so both are kept.
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


# A 9 by 8 grey thumbnail compared left to right gives 64 bits that describe where
# an image is lighter than itself, not what colour it is, so the same scene stored
# at two of this dataset's eleven resolutions still hashes to nearly the same bits.
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
            handle.convert("L").resize((DHASH_SIDE + 1, DHASH_SIDE), Image.BILINEAR),
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
    """Every pair whose hashes differ in at most ``max_distance`` bits.

    These are candidates only. A 64-bit hash collides on images that are merely
    flat and similar in tone, which this dataset has many of, so each pair must be
    confirmed against the pixels before it is called a duplicate.
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
                handle.convert("RGB").resize((side, side), Image.BILINEAR), dtype=np.int16
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
    """Divide one split into train and validation by group, balancing the rare classes.

    Two constraints pull against each other. Confirmed copies of the same photograph
    must land on the same side, or validation scores an image the model has trained
    on. And RO appears in only 86 of 1,430 images, so an unconstrained draw can leave
    validation with a handful of them and make its rare-class IoU meaningless. A
    grouped stratified fold satisfies both at once, which a random draw does neither.

    Each image is labelled with the rarest class it contains, measured on this data
    rather than assumed, so it is the scarce classes that decide the balance. Fold 0
    of ``n_splits`` becomes validation, making ``n_splits=5`` a 20 percent split.
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
    # Masking absent classes with a count above every real one turns "rarest class in
    # this image" into a single argmin, so every image gets exactly one stratum.
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
