"""Comparison tables and examples preserve dataset and training-budget boundaries."""

import numpy as np
import pytest

from scripts.evaluation.compare import aggregate, select_examples


def record(dataset, epochs):
    return {
        "dataset": dataset,
        "model": "linear",
        "config": {"epochs": epochs},
        "seed": 42,
        "metrics": {"accuracy": 0.8, "macro_f1": 0.79},
        "parameter_count": 10,
        "inference_seconds": 1.0,
        "training_seconds": 2.0,
        "git_commit": "test",
    }


def test_aggregation_separates_datasets_and_preliminary_runs():
    rows = aggregate([record("mnist", 50), record("fashion_mnist", 5), record("fashion_mnist", 50)])
    assert {(r["dataset"], r["epochs"], r["runs"]) for r in rows} == {
        ("mnist", 50, 1),
        ("fashion_mnist", 5, 1),
        ("fashion_mnist", 50, 1),
    }


def test_aggregation_rejects_duplicate_seeds_within_a_comparison():
    with pytest.raises(ValueError, match="duplicate seeds"):
        aggregate([record("mnist", 50), record("mnist", 50)])


def test_example_selection_takes_first_two_in_both_disagreement_directions():
    targets = np.zeros(9, dtype=int)
    baseline = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
    cnn = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0])
    selected = select_examples(targets, baseline, cnn)
    assert {key: value.tolist() for key, value in selected.items()} == {
        "CNN corrects baseline": [0, 1],
        "Both miss": [3, 4],
        "Baseline correct, CNN wrong": [5, 6],
    }


def test_example_selection_does_not_invent_missing_categories():
    correct = np.zeros(3, dtype=int)
    assert all(len(indices) == 0 for indices in select_examples(correct, correct, correct).values())
