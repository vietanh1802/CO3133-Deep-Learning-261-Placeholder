"""Run the Flickr8k exploratory analysis that backs the A3 dataset proposal.

Every number quoted in the A3 proposal comes from one run of this script. It adds no
measurement of its own: it calls :mod:`src.data.flickr8k` and :mod:`src.eda.flickr8k`
and writes the result to ``results/a3/eda/flickr8k/``.
"""

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: E402

from src.data.flickr8k import load_captions, load_split, tokenize  # noqa: E402
from src.eda.flickr8k import (  # noqa: E402
    caption_stats,
    captions_seen_in,
    check_image,
    judgement_summary,
    oov_rate,
    shared_captions,
)
from src.eda.suim import (  # noqa: E402
    find_duplicate_candidates,
    image_fingerprint,
    pixel_difference,
)
from src.plots.common import save_figure  # noqa: E402
from src.plots.format import apply  # noqa: E402

DATA_DIR = Path("data/flickr8k")
IMAGE_DIR = DATA_DIR / "Flicker8k_Dataset"
TEXT_DIR = DATA_DIR / "Flickr8k_text"
OUTPUT_DIR = Path("results/a3/eda/flickr8k")
SPLITS = ("train", "dev", "test")

# Same duplicate thresholds as the SUIM audit: hash candidates, then a pixel check.
HASH_DISTANCE = 12
PIXEL_TOLERANCE = 5.0
# Common input side of pretrained image encoders.
ENCODER_SIDE = 224


def main() -> None:
    """Measure images, captions, splits and pair quality, and write the evidence."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checks = [check_image(path) for path in sorted(IMAGE_DIR.glob("*.jpg"))]
    decoded = [c for c in checks if c.error is None]
    images = {c.name for c in decoded}
    captions, malformed = load_captions(TEXT_DIR / "Flickr8k.token.txt")
    split = {s: load_split(TEXT_DIR / f"Flickr_8k.{s}Images.txt") for s in SPLITS}
    split_of = {image: s for s in SPLITS for image in split[s]}
    train_captions = [c for image in split["train"] for c in captions[image]]

    def split_summary(name: str) -> dict:
        members = split[name]
        texts = [c for image in members if image in captions for c in captions[image]]
        shared = shared_captions(captions, members)
        return {
            "images": len(members),
            "duplicate_entries": len(members) - len(set(members)),
            "missing_image": sorted(set(members) - images),
            "missing_captions": sorted(set(members) - captions.keys()),
            **caption_stats(texts),
            "oov_rate_vs_train": None
            if name == "train"
            else round(oov_rate(train_captions, texts), 4),
            "shared_captions": {
                "captions": len(shared),
                "images_involved": len({i for owners in shared.values() for i in owners}),
                "examples": dict(list(shared.items())[:5]),
            },
            "identical_to_train_caption": None
            if name == "train"
            else seen_summary(captions_seen_in(train_captions, captions, members)),
        }

    def seen_summary(matches: list[tuple[str, str]]) -> dict:
        texts = list(dict.fromkeys(text for _, text in matches))
        return {
            "captions": len(matches),
            "images_involved": len({image for image, _ in matches}),
            "examples": texts[:5],
        }

    fingerprints = [
        image_fingerprint(IMAGE_DIR / c.name, split_of.get(c.name, "none")) for c in decoded
    ]
    candidates = find_duplicate_candidates(fingerprints, HASH_DISTANCE)
    confirmed = []
    for a, b, bits in candidates:
        score = pixel_difference(a.path, b.path)
        if score < PIXEL_TOLERANCE:
            confirmed.append(
                {
                    "images": [a.path.name, b.path.name],
                    "splits": [a.split, b.split],
                    "hash_bits": bits,
                    "pixel_difference": round(score, 2),
                }
            )

    expert_file = TEXT_DIR / "ExpertAnnotations.txt"
    judged = {
        line.split("\t")[0] for line in expert_file.read_text(encoding="utf-8").splitlines() if line
    }
    widths = [c.size[0] for c in decoded]
    heights = [c.size[1] for c in decoded]
    # Figure examples follow a fixed rule so that they are not hand-picked.
    examples = split["test"][:4]

    summary = {
        "images": {
            "files": len(checks),
            "decoded": len(decoded),
            "decode_errors": [(c.name, c.error) for c in checks if c.error],
            "bytes": sum(c.bytes for c in checks),
            "formats": dict(Counter(c.format for c in decoded)),
            "modes": dict(Counter(c.mode for c in decoded)),
            "width": {"min": min(widths), "median": statistics.median(widths), "max": max(widths)},
            "height": {
                "min": min(heights),
                "median": statistics.median(heights),
                "max": max(heights),
            },
            "distinct_sizes": len({c.size for c in decoded}),
            f"short_side_below_{ENCODER_SIDE}": sum(
                1 for c in decoded if min(c.size) < ENCODER_SIDE
            ),
        },
        "captions": {
            "images_with_captions": len(captions),
            "captions": sum(len(v) for v in captions.values()),
            "captions_per_image": dict(Counter(len(v) for v in captions.values())),
            "malformed_keys": malformed,
            "text_files_bytes": {p.name: p.stat().st_size for p in sorted(TEXT_DIR.glob("*.txt"))},
        },
        "pairing": {
            "images_without_captions": sorted(images - captions.keys()),
            "captions_without_image": sorted(captions.keys() - images),
            "images_in_no_split": len(images - split_of.keys()),
        },
        "splits": {s: split_summary(s) for s in SPLITS},
        # Digests of the official split files as downloaded, to reproduce the split.
        "split_files": {
            s: {
                "file": f"Flickr_8k.{s}Images.txt",
                "sha256": hashlib.sha256(
                    (TEXT_DIR / f"Flickr_8k.{s}Images.txt").read_bytes()
                ).hexdigest(),
            }
            for s in SPLITS
        },
        "split_overlap": {
            f"{a}&{b}": len(set(split[a]) & set(split[b]))
            for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
        },
        "duplicates": {
            "hash_distance": HASH_DISTANCE,
            "pixel_tolerance": PIXEL_TOLERANCE,
            "candidates_from_hash": len(candidates),
            "confirmed_by_pixels": confirmed,
            "crossing_splits": sum(1 for d in confirmed if d["splits"][0] != d["splits"][1]),
        },
        "judgements": {
            **judgement_summary(expert_file, TEXT_DIR / "CrowdFlowerAnnotations.txt"),
            "expert_images_by_split": dict(Counter(split_of.get(i, "none") for i in judged)),
        },
        "pair_examples": examples,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_caption_length(captions, split, OUTPUT_DIR / "caption_length.png")
    plot_top_words(train_captions, OUTPUT_DIR / "top_words.png")
    plot_resolutions(decoded, OUTPUT_DIR / "resolution_distribution.png")
    plot_pair_examples(examples, captions, OUTPUT_DIR / "pair_examples.png")
    if confirmed:
        plot_duplicate_examples(confirmed, OUTPUT_DIR / "duplicate_examples.png")
    total = summary["captions"]["captions"]
    print(f"{len(decoded)} images, {total} captions, evidence written to {OUTPUT_DIR}")


# Figures. apply() sets colours, fonts and size; save_figure() adds the grid.


def plot_caption_length(
    captions: dict[str, list[str]], split: dict[str, list[str]], path: Path
) -> None:
    """Show that caption length has the same distribution in every split."""
    apply()
    figure, axes = plt.subplots()
    for name, members in split.items():
        lengths = [len(tokenize(c)) for image in members for c in captions[image]]
        axes.hist(
            lengths, bins=range(1, max(lengths) + 2), density=True, histtype="step", label=name
        )
    axes.set(xlabel="Caption length (tokens)", ylabel="Share of captions")
    axes.set_title("Caption length by split")
    axes.legend()
    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def plot_top_words(train_captions: list[str], path: Path, top: int = 20) -> None:
    """Show which content words dominate the training captions (stop words removed)."""
    counts = Counter(
        word
        for caption in train_captions
        for word in tokenize(caption)
        if word.isalpha() and word not in ENGLISH_STOP_WORDS
    ).most_common(top)
    apply()
    figure, axes = plt.subplots()
    words, values = zip(*counts[::-1], strict=True)
    axes.bar_label(axes.barh(words, values), padding=2)
    axes.set_xlim(right=max(values) * 1.15)
    axes.set(xlabel="Occurrences in training captions", ylabel="Word")
    axes.set_title(f"{top} most frequent content words (stop words removed)")
    figure.set_size_inches(7.1, 4.0)
    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def plot_resolutions(decoded: list, path: Path) -> None:
    """Show image sizes: the long side is almost always 500 pixels."""
    sizes = Counter(c.size for c in decoded)
    apply()
    figure, axes = plt.subplots()
    axes.scatter(
        [w for w, _ in sizes],
        [h for _, h in sizes],
        s=[4 + n / 10 for n in sizes.values()],
        alpha=0.5,
    )
    axes.set(xlabel="Width (pixels)", ylabel="Height (pixels)")
    axes.set_title(
        f"{len(sizes)} distinct sizes across {len(decoded):,} images (marker area ~ count)"
    )
    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def plot_pair_examples(examples: list[str], captions: dict[str, list[str]], path: Path) -> None:
    """Show test images beside their five reference captions."""
    apply()
    figure, axes = plt.subplots(
        len(examples), 2, figsize=(7.1, 1.7 * len(examples)), gridspec_kw={"width_ratios": [1, 2.2]}
    )
    for row, image in zip(axes, examples, strict=True):
        with Image.open(IMAGE_DIR / image) as handle:
            row[0].imshow(handle.convert("RGB"))
        row[0].set_title(image, fontsize=6)
        row[1].text(
            0,
            0.5,
            "\n".join(f"{i + 1}. {c}" for i, c in enumerate(captions[image])),
            va="center",
            fontsize=6,
            wrap=True,
        )
        for axis in row:
            axis.axis("off")
    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


def plot_duplicate_examples(confirmed: list[dict], path: Path) -> None:
    """Show each confirmed duplicate pair with its splits and pixel difference."""
    apply()
    figure, axes = plt.subplots(
        len(confirmed), 2, figsize=(7.1, 2.4 * len(confirmed)), squeeze=False
    )
    for row, pair in zip(axes, confirmed, strict=True):
        for axis, image, name in zip(row, pair["images"], pair["splits"], strict=True):
            with Image.open(IMAGE_DIR / image) as handle:
                axis.imshow(handle.convert("RGB"))
            axis.set_title(f"{name}: {image}\npixel difference {pair['pixel_difference']}")
            axis.axis("off")
    figure.tight_layout()
    save_figure(figure, path)
    plt.close(figure)


if __name__ == "__main__":
    main()
