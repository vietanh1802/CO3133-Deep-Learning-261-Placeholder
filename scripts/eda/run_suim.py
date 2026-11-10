"""Run the SUIM exploratory analysis that backs the A2 dataset proposal.

Every number quoted in ``reports/a2_proposal.md`` comes from one run of this script.
It adds no measurement of its own: it calls :mod:`src.eda.suim`, applies the two
exclusion rules, and writes the result to ``results/a2/eda/suim/``.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

from src.config.segmentation import load_segmentation_config  # noqa: E402
from src.data.suim import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    SuimSegmentation,
    build_suim_loaders,
    find_pairs,
)
from src.eda.suim import (  # noqa: E402
    Fingerprint,
    PairCheck,
    check_pair,
    class_summary,
    find_duplicate_candidates,
    grouped_stratified_split,
    image_fingerprint,
    pixel_difference,
    usable_checks,
)
from src.plots.common import save_figure  # noqa: E402
from src.plots.format import COLORS, SEQUENTIAL, apply  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

DATA_DIR = Path("data/suim")
OUTPUT_DIR = Path("results/a2/eda/suim")
SPLIT_DIR = Path("results/a2/splits/suim")
SPLITS = ("train_val", "TEST")
# The example batch goes through the loader of the pretrained-model run.
BATCH_CONFIG = Path("configs/segmentation/suim/pretrained.yaml")

# Persist the split; regeneration must produce identical bytes.
SPLIT_SEED = 42
VAL_FOLDS = 5

# Confirmed duplicates differed by at most 9 hash bits across 1,200,475 pairs.
# Search up to 12 bits, then verify candidates against pixels.
HASH_DISTANCE = 12
PIXEL_TOLERANCE = 5.0


def duplicate_groups(
    fingerprints: list[Fingerprint],
) -> tuple[int, int, list[list[str]]]:
    """Group images that are confirmed copies of one another."""
    parent = {f"{f.split}/{f.stem}": f"{f.split}/{f.stem}" for f in fingerprints}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    candidates = find_duplicate_candidates(fingerprints, HASH_DISTANCE)
    confirmed = [
        (a, b) for a, b, _ in candidates if pixel_difference(a.path, b.path) < PIXEL_TOLERANCE
    ]
    for a, b in confirmed:
        left, right = root(f"{a.split}/{a.stem}"), root(f"{b.split}/{b.stem}")
        if left != right:
            parent[left] = right
    members: dict[str, list[str]] = {}
    for key in parent:
        members.setdefault(root(key), []).append(key)
    return len(candidates), len(confirmed), [sorted(g) for g in members.values() if len(g) > 1]


def label_structure(checks: list[PairCheck]) -> dict:
    """How many classes share an image, and which classes those are.

    A classification dataset has one label per sample and neither question exists.
    Here the answers steer section 5: a class pair that shares many images is a pair
    the confusion matrix has a chance to confuse.
    """
    present = np.array([[count > 0 for count in c.class_pixels] for c in checks], dtype=np.int32)
    values, counts = np.unique(present.sum(axis=1), return_counts=True)
    together = present.T @ present
    return {
        "classes_per_image": {str(v): int(n) for v, n in zip(values, counts, strict=True)},
        "co_occurrence": {
            CLASS_NAMES[i]: {CLASS_NAMES[j]: int(together[i, j]) for j in range(NUM_CLASSES)}
            for i in range(NUM_CLASSES)
        },
    }


def median_by_off_code(checks: list[PairCheck]) -> PairCheck:
    """The check with the median share of off-code mask pixels, ties broken by name."""
    ordered = sorted(checks, key=lambda c: (c.off_code_pixels / c.total_pixels, c.stem))
    return ordered[len(ordered) // 2]


def off_code_range(checks: list[PairCheck]) -> list[float] | None:
    """Smallest and largest share of off-code pixels among masks that have any."""
    shares = [c.off_code_pixels / c.total_pixels for c in checks if c.off_code_pixels]
    return [round(min(shares), 4), round(max(shares), 4)] if shares else None


def example_batch(config_path: Path) -> dict:
    """Draw one training batch with the sampler, transforms and collate function of a run."""
    config = load_segmentation_config(config_path)
    seed_everything(config.seed)
    loader = build_suim_loaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        seed=config.seed,
        image_height=config.image_height,
        image_width=config.image_width,
        augment=config.augment,
    ).train
    dataset = loader.dataset
    assert isinstance(dataset, SuimSegmentation)
    assert loader.batch_sampler is not None
    indices = next(iter(loader.batch_sampler))
    images, masks = loader.collate_fn([dataset[i] for i in indices])
    # The flip is random, so an unaugmented copy tells which samples it hit.
    plain = SuimSegmentation(
        config.data_dir, "train", image_size=(config.image_height, config.image_width)
    )
    return {
        "images": images,
        "masks": masks,
        "display": dataset.denormalize(images),
        "stems": [dataset.stems[i] for i in indices],
        "flipped": [not torch.equal(images[k], plain[i][0]) for k, i in enumerate(indices)],
    }


def main() -> None:
    """Measure the dataset, apply both exclusion rules, and write the evidence."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    found, checked, kept = {}, {}, {}
    for split in SPLITS:
        found[split] = find_pairs(DATA_DIR / split)
        checked[split] = [check_pair(image, mask) for image, mask in found[split].pairs]
        kept[split] = usable_checks(checked[split])

    fingerprints = [
        image_fingerprint(DATA_DIR / split / "images" / f"{c.stem}.jpg", split)
        for split in SPLITS
        for c in kept[split]
    ]
    candidates, confirmed, groups = duplicate_groups(fingerprints)
    crossing = [g for g in groups if len({m.split("/")[0] for m in g}) > 1]
    # Preserve TEST; drop matching train_val images.
    contaminated = {m.split("/")[1] for g in crossing for m in g if m.startswith("train_val/")}
    kept["train_val"] = [c for c in kept["train_val"] if c.stem not in contaminated]

    assignment = grouped_stratified_split(
        kept["train_val"],
        # Cross-TEST duplicates are removed; group the remaining train_val images.
        [[m.split("/")[1] for m in g] for g in groups if not any(m.startswith("TEST/") for m in g)],
        n_splits=VAL_FOLDS,
        seed=SPLIT_SEED,
    )
    # Figure examples follow fixed rules so that they are not hand-picked.
    paths = {
        split: {image.stem: (image, mask) for image, mask in found[split].pairs} for split in SPLITS
    }
    rejected = [c for c in checked["train_val"] if c.off_code_pixels]
    # One rejected mask of each kind: same size as its image, and taller than it.
    off_code_examples = [
        median_by_off_code([c for c in rejected if c.size_matches == matches])
        for matches in (True, False)
    ]
    # The most and the least similar copies of a test image in the training pool.
    duplicate_pairs = []
    for group in crossing:
        copy = next(m.split("/")[1] for m in group if m.startswith("train_val/"))
        original = next(m.split("/")[1] for m in group if m.startswith("TEST/"))
        score = pixel_difference(paths["train_val"][copy][0], paths["TEST"][original][0])
        duplicate_pairs.append((score, copy, original))
    duplicate_pairs.sort()
    duplicate_examples = [duplicate_pairs[0], duplicate_pairs[-1]]

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    by_stem = {c.stem: c for c in kept["train_val"]}
    halves, digests = {}, {}
    for name, members in (("train", assignment.train), ("val", assignment.val)):
        path = SPLIT_DIR / f"{name}.txt"
        # Fix newlines for reproducible cross-platform digests.
        path.write_text("\n".join(members) + "\n", encoding="utf-8", newline="\n")
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        halves[name] = [by_stem[stem] for stem in members]
    # Drawn after the split files are written, because the loader reads them.
    batch = example_batch(BATCH_CONFIG)

    summary = {
        "splits": {
            split: {
                "pairs_on_disk": len(found[split].pairs),
                "image_bytes": sum(image.stat().st_size for image, _ in found[split].pairs),
                "mask_bytes": sum(mask.stat().st_size for _, mask in found[split].pairs),
                "images_without_mask": found[split].images_without_mask,
                "masks_without_image": found[split].masks_without_image,
                "decode_errors": sum(1 for c in checked[split] if c.error),
                "size_mismatches": sum(
                    1 for c in checked[split] if not c.error and not c.size_matches
                ),
                "off_code_masks": sum(1 for c in checked[split] if c.off_code_pixels > 0),
                # Clean masks have none, so any cut-off below the minimum gives the same result.
                "off_code_share_range": off_code_range(checked[split]),
                "usable_pairs": len(kept[split]),
                "labelled_pixels": sum(c.total_pixels for c in kept[split]),
                "resolutions": {
                    f"{w}x{h}": n
                    for (w, h), n in Counter(
                        c.image_size for c in kept[split] if c.image_size is not None
                    ).most_common()
                },
                "classes": [
                    {
                        "name": s.name,
                        "images": s.image_count,
                        "pixels": s.pixel_count,
                        "pixel_share": round(s.pixel_share, 6),
                        "median_area_when_present": round(s.median_area_when_present, 6),
                    }
                    for s in class_summary(kept[split])
                ],
                **label_structure(kept[split]),
            }
            for split in SPLITS
        },
        "duplicates": {
            "hash_distance": HASH_DISTANCE,
            "pixel_tolerance": PIXEL_TOLERANCE,
            "candidates_from_hash": candidates,
            "confirmed_by_pixels": confirmed,
            "groups": groups,
            "groups_crossing_splits": crossing,
            "dropped_from_train_val": sorted(contaminated),
        },
        "split": {
            "seed": assignment.seed,
            "val_fold_of": assignment.n_splits,
            "group_count": assignment.group_count,
            "stratum_sizes": assignment.stratum_sizes,
            "sha256_prefix": digests,
            "halves": {
                name: {
                    "images": len(checks),
                    "classes": {
                        s.name: {"images": s.image_count, "pixel_share": round(s.pixel_share, 4)}
                        for s in class_summary(checks)
                    },
                }
                for name, checks in halves.items()
            },
        },
        "usable_pairs_total": sum(len(v) for v in kept.values()),
        "audit_examples": {
            "off_code_masks": [
                {
                    "stem": c.stem,
                    "image_size": c.image_size,
                    "mask_size": c.mask_size,
                    "off_code_share": round(c.off_code_pixels / c.total_pixels, 4),
                }
                for c in off_code_examples
            ],
            "cross_split_pairs": [
                {"train_val": copy, "TEST": original, "pixel_difference": round(score, 2)}
                for score, copy, original in duplicate_examples
            ],
        },
        "example_batch": {
            "config": BATCH_CONFIG.as_posix(),
            "stems": batch["stems"],
            "flipped": batch["flipped"],
            "image_shape": list(batch["images"].shape),
            "image_dtype": str(batch["images"].dtype).removeprefix("torch."),
            "image_range": [
                round(batch["images"].min().item(), 2),
                round(batch["images"].max().item(), 2),
            ],
            "mask_shape": list(batch["masks"].shape),
            "mask_dtype": str(batch["masks"].dtype).removeprefix("torch."),
            "mask_classes": [CLASS_NAMES[i] for i in batch["masks"].unique().tolist()],
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    stats = class_summary(kept["train_val"])
    plot_class_distribution(stats, len(kept["train_val"]), OUTPUT_DIR / "class_distribution.png")
    plot_region_area(kept["train_val"], OUTPUT_DIR / "region_area_distribution.png")
    plot_resolutions(kept["train_val"] + kept["TEST"], OUTPUT_DIR / "resolution_distribution.png")
    plot_off_code_examples(
        [paths["train_val"][c.stem] for c in off_code_examples],
        OUTPUT_DIR / "off_code_examples.png",
    )
    plot_duplicate_examples(
        [
            (paths["train_val"][copy][0], paths["TEST"][original][0], score)
            for score, copy, original in duplicate_examples
        ],
        OUTPUT_DIR / "duplicate_examples.png",
    )
    plot_example_batch(batch, OUTPUT_DIR / "example_batch.png")
    print(f"{summary['usable_pairs_total']} usable pairs, evidence written to {OUTPUT_DIR}")
    print(f"split {len(assignment.train)} train / {len(assignment.val)} val in {SPLIT_DIR}")


# Figures. apply() sets colours, fonts and size; save_figure() adds the grid.


def plot_class_distribution(stats, image_total: int, path: Path) -> None:
    """Show that counting by image and counting by pixel rank the classes differently."""
    apply()
    figure, axes = plt.subplots()
    position = np.arange(NUM_CLASSES)
    width = 0.4
    images = [s.image_count / image_total for s in stats]
    pixels = [s.pixel_share for s in stats]
    axes.bar(position - width / 2, images, width, label="Images containing the class")
    axes.bar(position + width / 2, pixels, width, label="Share of all pixels")
    axes.set_xticks(position, CLASS_NAMES)
    axes.set(xlabel="Class", ylabel="Proportion of train and validation pool")
    axes.set_title("Class frequency by image and by pixel", pad=20)
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def plot_region_area(checks: list[PairCheck], path: Path) -> None:
    """Separate a class that is rare because it is absent from one that is small."""
    areas = [
        [c.class_pixels[i] / c.total_pixels for c in checks if c.class_pixels[i]]
        for i in range(NUM_CLASSES)
    ]
    apply()
    figure, axes = plt.subplots()
    axes.boxplot(areas, tick_labels=list(CLASS_NAMES), showfliers=False)
    axes.set(xlabel="Class", ylabel="Fraction of image where present")
    axes.set_title("Region area per class")
    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def plot_resolutions(checks: list[PairCheck], path: Path) -> None:
    """Show how far the data departs from the four resolutions the paper names."""
    counts = Counter(c.image_size for c in checks if c.image_size is not None).most_common()
    labels = [f"{w}×{h}" for (w, h), _ in counts][::-1]
    values = [n for _, n in counts][::-1]
    apply()
    figure, axes = plt.subplots()
    axes.bar_label(axes.barh(labels, values), padding=2)
    axes.set_xscale("log")
    # Leave room on the log axis for the count printed after the longest bar.
    axes.set_xlim(right=max(values) * 3)
    axes.set(xlabel="Number of images (log scale)", ylabel="Resolution (width × height)")
    axes.set_title(f"{len(counts)} native resolutions across {len(checks):,} pairs")
    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def load_rgb(file: Path) -> np.ndarray:
    with Image.open(file) as handle:
        return np.asarray(handle.convert("RGB"))


def plot_off_code_examples(pairs: list[tuple[Path, Path]], path: Path) -> None:
    """Show masks that look clean but hold colours outside the eight class codes."""
    apply()
    figure, axes = plt.subplots(len(pairs), 3, figsize=(7.1, 2.1 * len(pairs)))
    for row, (image_path, mask_path) in zip(axes, pairs, strict=True):
        image, mask = load_rgb(image_path), load_rgb(mask_path)
        off_code = ~((mask == 0) | (mask == 255)).all(axis=2)
        row[0].imshow(image)
        row[0].set_title(f"{image_path.stem} ({image.shape[1]}×{image.shape[0]})")
        row[1].imshow(mask)
        row[1].set_title(f"Mask as stored ({mask.shape[1]}×{mask.shape[0]})")
        row[2].imshow(off_code, cmap=ListedColormap(["white", COLORS[3]]), vmin=0, vmax=1)
        row[2].set_title(f"Off-code pixels ({off_code.mean():.0%})")
        for axis in row:
            axis.axis("off")
    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def plot_duplicate_examples(pairs: list[tuple[Path, Path, float]], path: Path) -> None:
    """Show training-pool copies of test images beside the pixel difference that confirms them."""
    apply()
    figure, axes = plt.subplots(len(pairs), 3, figsize=(7.1, 2.0 * len(pairs)))
    for row, (copy_path, original_path, score) in zip(axes, pairs, strict=True):
        copy, original = load_rgb(copy_path), load_rgb(original_path)
        height, width = original.shape[:2]
        resized = Image.fromarray(copy).resize((width, height), Image.Resampling.BILINEAR)
        gap = np.abs(np.asarray(resized, dtype=np.int16) - original).mean(axis=2)
        row[0].imshow(copy)
        row[0].set_title(f"Training pool\n{copy_path.stem} ({copy.shape[1]}×{copy.shape[0]})")
        row[1].imshow(original)
        row[1].set_title(f"Test set\n{original_path.stem} ({width}×{height})")
        row[2].imshow(gap, cmap=SEQUENTIAL, vmin=0, vmax=255)
        row[2].set_title(f"Difference\naudit score {score:.1f}")
        for axis in row:
            axis.axis("off")
    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


# SUIM stores class i as the RGB colour whose bits are those of i, e.g. RO = 100 = red.
SUIM_COLORS = np.array(
    [[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(NUM_CLASSES)], dtype=np.float32
)


def plot_example_batch(batch: dict, path: Path) -> None:
    """Show one preprocessed training batch: input, decoded mask, and the two overlaid."""
    rows = ("Input (denormalized)", "Class mask", "Overlay")
    columns = len(batch["stems"])
    apply()
    figure, axes = plt.subplots(len(rows), columns, figsize=(7.1, 5.0), squeeze=False)
    for column, (image, mask, stem, flipped) in enumerate(
        zip(batch["display"], batch["masks"], batch["stems"], batch["flipped"], strict=True)
    ):
        image = image.permute(1, 2, 0).numpy()
        colours = SUIM_COLORS[mask.numpy()]
        # Background water stays unblended so that the labelled objects stand out.
        foreground = (mask.numpy() > 0)[..., None]
        overlay = np.where(foreground, 0.5 * image + 0.5 * colours, image)
        for row, picture in enumerate((image, colours, overlay)):
            axis = axes[row, column]
            axis.imshow(picture)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(rows[row])
        axes[0, column].set_title(f"{stem}\n{'flipped' if flipped else 'not flipped'}")
    figure.legend(
        handles=[
            Patch(facecolor=colour, edgecolor="0.5", label=name)
            for name, colour in zip(CLASS_NAMES, SUIM_COLORS, strict=True)
        ],
        loc="lower center",
        ncol=NUM_CLASSES,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(figure, path)
    plt.close(figure)


if __name__ == "__main__":
    main()
