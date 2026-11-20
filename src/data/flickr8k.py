"""Flickr8k caption and split files."""

from pathlib import Path

IMAGE_SUFFIX = ".jpg"


def load_captions(token_file: str | Path) -> tuple[dict[str, list[str]], list[str]]:
    """Read ``Flickr8k.token.txt`` into captions per image filename.

    Each line is ``<image>#<n>\\t<caption>``. Keys whose image name does not end in
    ``.jpg`` (the release contains ``2258277193_586949ec62.jpg.1``) are returned in
    the second list and not loaded.
    """
    captions: dict[str, list[str]] = {}
    malformed: list[str] = []
    for line in Path(token_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, caption = line.split("\t", 1)
        image = key.rsplit("#", 1)[0]
        if not image.endswith(IMAGE_SUFFIX):
            malformed.append(key)
            continue
        captions.setdefault(image, []).append(caption.strip())
    return captions, malformed


def load_split(split_file: str | Path) -> list[str]:
    """Read one split file: one image filename per line."""
    lines = Path(split_file).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def tokenize(caption: str) -> list[str]:
    """Lowercase and split on whitespace, dropping tokens with no letter or digit.

    Captions are already space-tokenized in the release ("A dog 's ball ."), so
    this keeps "'s" and drops the standalone "." and ",".
    """
    return [token for token in caption.lower().split() if any(c.isalnum() for c in token)]
