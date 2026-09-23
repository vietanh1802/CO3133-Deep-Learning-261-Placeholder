import json
from pathlib import Path

import numpy as np
import pytest

from src.data.eda import (
    as_image_batch,
    compute_class_mean_images,
    compute_class_similarity,
    format_summary_lines,
    most_similar_class_pairs,
    profile_dataset,
    summarize_class_balance,
    summarize_image_shape,
    summarize_intensity,
    summary_to_dict,
    write_summary_json,
)

CLASS_NAMES = ("dark", "bright", "striped")


def _toy_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Build a 6-image, 3-class uint8 dataset with known statistics.

    Class 0 is blank, class 1 is fully bright, and class 2 has one bright row, so every
    per-class number below can be verified by hand.
    """
    images = np.zeros((6, 4, 4), dtype=np.uint8)
    images[2:4] = 255
    images[4:6, 0, :] = 255
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    return images, labels


def test_as_image_batch_adds_a_channel_axis_to_grayscale() -> None:
    assert as_image_batch(np.zeros((2, 4, 4), dtype=np.uint8)).shape == (2, 4, 4, 1)
    assert as_image_batch(np.zeros((2, 4, 4, 3), dtype=np.uint8)).shape == (2, 4, 4, 3)


@pytest.mark.parametrize(
    ("images", "message"),
    [
        (np.zeros((4, 4), dtype=np.uint8), "must have shape"),
        (np.zeros((0, 4, 4), dtype=np.uint8), "must not be empty"),
    ],
)
def test_as_image_batch_rejects_unusable_input(images: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        as_image_batch(images)


def test_summarize_image_shape_reports_geometry_and_value_range() -> None:
    images, _ = _toy_dataset()
    shape = summarize_image_shape(images)

    assert (shape.num_samples, shape.height, shape.width, shape.channels) == (6, 4, 4, 1)
    assert shape.dtype == "uint8"
    assert (shape.min_value, shape.max_value) == (0.0, 255.0)


def test_summarize_class_balance_counts_every_class() -> None:
    _, labels = _toy_dataset()
    balance = summarize_class_balance(labels, CLASS_NAMES)

    assert balance.counts == (2, 2, 2)
    assert balance.proportions == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert balance.imbalance_ratio == pytest.approx(1.0)


def test_summarize_class_balance_flags_the_rarest_and_most_common_class() -> None:
    balance = summarize_class_balance(np.array([0, 1, 1, 1, 2, 2]), CLASS_NAMES)

    assert balance.counts == (1, 3, 2)
    assert balance.imbalance_ratio == pytest.approx(3.0)
    assert balance.rarest_class == "dark"
    assert balance.most_common_class == "bright"


def test_summarize_class_balance_rejects_labels_outside_the_class_list() -> None:
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        summarize_class_balance(np.array([0, 3]), CLASS_NAMES)


def test_summarize_intensity_matches_hand_computed_per_class_values() -> None:
    images, labels = _toy_dataset()
    intensity = summarize_intensity(images, labels, num_classes=len(CLASS_NAMES))

    # Blank, fully bright, and one bright row out of four.
    assert intensity.per_class_mean_intensity == pytest.approx((0.0, 255.0, 63.75))
    assert intensity.per_class_foreground_fraction == pytest.approx((0.0, 1.0, 0.25))
    assert intensity.mean == pytest.approx(np.mean(images))
    assert sum(intensity.histogram) == images.size


def test_summarize_intensity_rejects_mismatched_labels() -> None:
    images, _ = _toy_dataset()
    with pytest.raises(ValueError, match="same number of samples"):
        summarize_intensity(images, np.array([0, 1]), num_classes=3)


def test_compute_class_mean_images_averages_within_each_class() -> None:
    images, labels = _toy_dataset()
    means = compute_class_mean_images(images, labels, num_classes=len(CLASS_NAMES))

    assert means.shape == (3, 4, 4, 1)
    assert np.all(means[0] == 0.0)
    assert np.all(means[1] == 255.0)
    assert np.all(means[2][0] == 255.0)
    assert np.all(means[2][1:] == 0.0)


def test_compute_class_similarity_is_symmetric_with_a_unit_diagonal() -> None:
    images, labels = _toy_dataset()
    means = compute_class_mean_images(images, labels, num_classes=len(CLASS_NAMES))
    similarity = compute_class_similarity(means)

    assert similarity.shape == (3, 3)
    assert np.allclose(similarity, similarity.T)
    assert np.allclose(np.diagonal(similarity), 1.0)
    assert np.all(similarity <= 1.0 + 1e-9)
    assert np.all(similarity >= -1.0 - 1e-9)


def test_compute_class_similarity_scores_identical_prototypes_as_identical() -> None:
    shared = np.full((4, 4, 1), 200.0)
    means = np.stack([shared, shared, np.zeros((4, 4, 1))])

    similarity = compute_class_similarity(means)

    assert similarity[0, 1] == pytest.approx(1.0)


def test_most_similar_class_pairs_is_sorted_and_excludes_self_pairs() -> None:
    images, labels = _toy_dataset()
    profile = profile_dataset(images, labels, CLASS_NAMES, dataset="toy", split="train")

    pairs = most_similar_class_pairs(profile, top_k=3)

    assert len(pairs) == 3
    assert all(first != second for first, second, _ in pairs)
    assert [similarity for _, _, similarity in pairs] == sorted(
        (similarity for _, _, similarity in pairs), reverse=True
    )


def test_most_similar_class_pairs_rejects_a_non_positive_top_k() -> None:
    images, labels = _toy_dataset()
    profile = profile_dataset(images, labels, CLASS_NAMES, dataset="toy", split="train")

    with pytest.raises(ValueError, match="top_k must be positive"):
        most_similar_class_pairs(profile, top_k=0)


def test_summary_to_dict_is_json_serializable() -> None:
    images, labels = _toy_dataset()
    profile = profile_dataset(images, labels, CLASS_NAMES, dataset="toy", split="train")

    summary = json.loads(json.dumps(summary_to_dict(profile)))

    assert summary["dataset"] == "toy"
    assert summary["split"] == "train"
    assert summary["class_balance"]["class_names"] == list(CLASS_NAMES)
    assert np.allclose(summary["class_similarity"], profile.class_similarity)


def test_write_summary_json_creates_missing_directories(tmp_path: Path) -> None:
    images, labels = _toy_dataset()
    profile = profile_dataset(images, labels, CLASS_NAMES, dataset="toy", split="train")

    path = write_summary_json(profile, tmp_path / "nested" / "summary.json")

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == summary_to_dict(profile)


def test_format_summary_lines_covers_every_class() -> None:
    images, labels = _toy_dataset()
    profile = profile_dataset(images, labels, CLASS_NAMES, dataset="toy", split="train")

    text = "\n".join(format_summary_lines(profile))

    assert "Dataset: toy (train split)" in text
    for name in CLASS_NAMES:
        assert name in text
    assert "Imbalance ratio (max/min): 1.00" in text
