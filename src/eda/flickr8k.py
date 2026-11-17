"""Flickr8k image checks, caption statistics, and human pair judgements."""

import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.data.flickr8k import tokenize


@dataclass(frozen=True)
class ImageCheck:
    """One image's file size and decoded properties, or the error that stopped decoding."""

    name: str
    bytes: int
    size: tuple[int, int] | None
    mode: str | None
    format: str | None
    error: str | None = None


def check_image(path: Path) -> ImageCheck:
    """Decode one image fully and report its file size, pixel size, mode and format."""
    file_bytes = path.stat().st_size
    try:
        with Image.open(path) as handle:
            # Force full decoding to detect truncated files.
            handle.load()
            return ImageCheck(
                name=path.name,
                bytes=file_bytes,
                size=handle.size,
                mode=handle.mode,
                format=handle.format,
            )
    except (OSError, ValueError) as exc:
        return ImageCheck(
            name=path.name,
            bytes=file_bytes,
            size=None,
            mode=None,
            format=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def caption_stats(captions: Iterable[str], top: int = 20) -> dict:
    """Length in tokens, vocabulary size, and most frequent tokens of a caption set."""
    tokens = [tokenize(caption) for caption in captions]
    lengths = [len(t) for t in tokens]
    counts = Counter(word for t in tokens for word in t)
    return {
        "captions": len(lengths),
        "length": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "mean": round(statistics.mean(lengths), 2),
            "max": max(lengths),
        },
        "at_most_3_tokens": sum(1 for n in lengths if n <= 3),
        "vocabulary": len(counts),
        "words_seen_once": sum(1 for n in counts.values() if n == 1),
        "top_words": counts.most_common(top),
    }


def oov_rate(train_captions: Iterable[str], other_captions: Iterable[str]) -> float:
    """Share of tokens in other_captions that never occur in train_captions."""
    vocabulary = {word for caption in train_captions for word in tokenize(caption)}
    tokens = [word for caption in other_captions for word in tokenize(caption)]
    return sum(1 for word in tokens if word not in vocabulary) / len(tokens)


def shared_captions(
    captions: dict[str, list[str]], image_ids: Iterable[str]
) -> dict[str, list[str]]:
    """Captions that are identical, after tokenizing, for two or more images.

    Notes:
        In retrieval such a caption has several correct images, but the metric
        accepts only the one it was written for.
    """
    owners: dict[str, set[str]] = {}
    for image in image_ids:
        for caption in captions[image]:
            owners.setdefault(" ".join(tokenize(caption)), set()).add(image)
    return {text: sorted(images) for text, images in sorted(owners.items()) if len(images) > 1}


def captions_seen_in(
    reference: Iterable[str], captions: dict[str, list[str]], image_ids: Iterable[str]
) -> list[tuple[str, str]]:
    """Captions of image_ids that are identical, after tokenizing, to a reference caption.

    Returns:
        One (image, tokenized caption) entry per matching caption, in input order.

    Notes:
        With the training captions as reference, a match is a held-out caption whose
        exact text was seen during training, although written for a different image.
    """
    seen = {" ".join(tokenize(caption)) for caption in reference}
    return [
        (image, text)
        for image in image_ids
        for text in (" ".join(tokenize(caption)) for caption in captions[image])
        if text in seen
    ]


def judgement_summary(expert_file: Path, crowd_file: Path) -> dict:
    """Summarise the human image-caption judgements released with Flickr8k.

    Both files list ``<image>\\t<image>#<n>\\t...``. A pair is "own" when the
    caption was written for that image and "other" when it was written for a
    different one. Expert scores run from 1 (unrelated) to 4 (describes the image),
    three experts per pair. CrowdFlower columns are yes-share, yes count, no count.
    Own pairs measure whether captions match their images; other pairs measure how
    often a caption written for another image still fits.
    """

    def rows(path: Path) -> list[tuple[str, bool, list[float]]]:
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                image, caption_id, *values = line.split("\t")
                own = caption_id.rsplit("#", 1)[0] == image
                out.append((caption_id, own, [float(v) for v in values]))
        return out

    expert = rows(expert_file)
    own_scores = [v for _, own, v in expert if own]
    other_scores = [v for _, own, v in expert if not own]
    crowd = rows(crowd_file)
    own_crowd = [(caption_id, v) for caption_id, own, v in crowd if own]
    other_crowd = [v for _, own, v in crowd if not own]
    return {
        "expert": {
            "pairs": len(expert),
            "own_caption_pairs": len(own_scores),
            "own_caption_rounded_mean_score": dict(
                sorted(Counter(round(statistics.mean(v)) for v in own_scores).items())
            ),
            "other_caption_pairs": len(other_scores),
            "other_caption_rounded_mean_score": dict(
                sorted(Counter(round(statistics.mean(v)) for v in other_scores).items())
            ),
            "other_caption_all_experts_4": sum(1 for v in other_scores if min(v) == 4),
        },
        "crowdflower": {
            "pairs": len(crowd),
            "own_caption_pairs": len(own_crowd),
            # Own captions that most judges rejected: candidate mismatched pairs.
            "own_caption_without_majority_yes": sorted(
                caption_id for caption_id, v in own_crowd if v[0] <= 0.5
            ),
            "other_caption_pairs": len(other_crowd),
            "other_caption_majority_yes": sum(1 for v in other_crowd if v[0] > 0.5),
        },
    }
