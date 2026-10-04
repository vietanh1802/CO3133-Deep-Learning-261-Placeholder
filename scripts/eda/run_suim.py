"""Run the SUIM exploratory analysis that backs the A2 dataset proposal.

Every number quoted in ``reports/a2_proposal.md`` comes from one run of this script.
It adds no measurement of its own: it calls :mod:`src.data.suim`, applies the two
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

from src.data.suim import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    Fingerprint,
    PairCheck,
    check_pair,
    class_summary,
    find_duplicate_candidates,
    find_pairs,
    grouped_stratified_split,
    image_fingerprint,
    pixel_difference,
    usable_checks,
)

DATA_DIR = Path("data/suim")
OUTPUT_DIR = Path("results/a2/eda/suim")
SPLIT_DIR = Path("results/a2/splits/suim")
SPLITS = ("train_val", "TEST")

# The split is a decision, not a measurement, so it is written to disk and committed.
# Regenerating it from this seed must reproduce the files byte for byte.
SPLIT_SEED = 42
VAL_FOLDS = 5

# Calibrated, not guessed: over all 1,200,475 pairs no duplicate confirmed against
# the pixels differed in more than 9 of the 64 hash bits, so 12 searches past the
# evidence. The hash alone is far too loose on water scenes, which is why every
# candidate is then compared pixel by pixel.
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


def plot_class_distribution(stats, image_total: int, path: Path) -> None:
    """Show that counting by image and counting by pixel rank the classes differently."""
    figure, axes = plt.subplots(figsize=(9, 4))
    position = np.arange(NUM_CLASSES)
    axes.bar(position - 0.2, [s.image_count / image_total for s in stats], 0.4,
             label="images containing the class")
    axes.bar(position + 0.2, [s.pixel_share for s in stats], 0.4, label="share of all pixels")
    axes.set_xticks(position, CLASS_NAMES)
    axes.set_ylabel("share of train_val")
    axes.set_title("Class frequency by image and by pixel")
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_region_area(checks: list[PairCheck], path: Path) -> None:
    """Separate a class that is rare because it is absent from one that is small."""
    areas = [
        [c.class_pixels[i] / c.total_pixels for c in checks if c.class_pixels[i]]
        for i in range(NUM_CLASSES)
    ]
    figure, axes = plt.subplots(figsize=(9, 4))
    axes.boxplot(areas, tick_labels=list(CLASS_NAMES), showfliers=False)
    axes.set_ylabel("share of the image, where the class is present")
    axes.set_title("How much of an image a class covers when it appears")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_resolutions(checks: list[PairCheck], path: Path) -> None:
    """Show how far the data departs from the four resolutions the paper names."""
    counts = Counter(c.image_size for c in checks).most_common()
    figure, axes = plt.subplots(figsize=(9, 4))
    axes.barh([f"{w}x{h}" for (w, h), _ in counts][::-1], [n for _, n in counts][::-1])
    axes.set_xscale("log")
    axes.set_xlabel("images (log scale)")
    axes.set_title(f"{len(counts)} resolutions across {len(checks)} pairs")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


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
    # TEST is the published benchmark and stays whole, so the train_val twin goes.
    contaminated = {m.split("/")[1] for g in crossing for m in g if m.startswith("train_val/")}
    kept["train_val"] = [c for c in kept["train_val"] if c.stem not in contaminated]

    assignment = grouped_stratified_split(
        kept["train_val"],
        # Only groups wholly inside train_val are left to keep together; the ones that
        # crossed into TEST were already resolved by dropping their train_val member.
        [[m.split("/")[1] for m in g] for g in groups if not any(m.startswith("TEST/") for m in g)],
        n_splits=VAL_FOLDS,
        seed=SPLIT_SEED,
    )
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    by_stem = {c.stem: c for c in kept["train_val"]}
    halves, digests = {}, {}
    for name, members in (("train", assignment.train), ("val", assignment.val)):
        path = SPLIT_DIR / f"{name}.txt"
        # newline= is explicit so a Windows run and a Linux run write the same bytes,
        # which is what makes the digest below worth quoting.
        path.write_text("\n".join(members) + "\n", encoding="utf-8", newline="\n")
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        halves[name] = [by_stem[stem] for stem in members]

    summary = {
        "splits": {
            split: {
                "pairs_on_disk": len(found[split].pairs),
                "images_without_mask": found[split].images_without_mask,
                "masks_without_image": found[split].masks_without_image,
                "decode_errors": sum(1 for c in checked[split] if c.error),
                "size_mismatches": sum(
                    1 for c in checked[split] if not c.error and not c.size_matches
                ),
                "off_code_masks": sum(1 for c in checked[split] if c.off_code_pixels > 0),
                "usable_pairs": len(kept[split]),
                "labelled_pixels": sum(c.total_pixels for c in kept[split]),
                "resolutions": {f"{w}x{h}": n for (w, h), n in
                                Counter(c.image_size for c in kept[split]).most_common()},
                "classes": [
                    {"name": s.name, "images": s.image_count, "pixels": s.pixel_count,
                     "pixel_share": round(s.pixel_share, 6),
                     "median_area_when_present": round(s.median_area_when_present, 6)}
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
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    stats = class_summary(kept["train_val"])
    plot_class_distribution(stats, len(kept["train_val"]), OUTPUT_DIR / "class_distribution.png")
    plot_region_area(kept["train_val"], OUTPUT_DIR / "region_area_distribution.png")
    plot_resolutions(kept["train_val"] + kept["TEST"], OUTPUT_DIR / "resolution_distribution.png")
    print(f"{summary['usable_pairs_total']} usable pairs, evidence written to {OUTPUT_DIR}")
    print(f"split {len(assignment.train)} train / {len(assignment.val)} val in {SPLIT_DIR}")


if __name__ == "__main__":
    main()
