"""Comparison tables and examples preserve dataset and training-budget boundaries."""

import json

import numpy as np
import pytest

from scripts.evaluation.compare import aggregate, load_records, select_examples


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


@pytest.mark.parametrize("dataset", ["fashion_mnist", "mnist", "cifar10"])
def test_comparison_uses_spatial_cnn_when_legacy_run_is_also_present(tmp_path, dataset):
    legacy = {
        **record(dataset, 50),
        "model": "cnn",
        "name": f"{dataset}_cnn_seed42",
        "parameter_count": 20,
        "metrics": {"accuracy": 0.6, "macro_f1": 0.59},
    }
    spatial = {
        **legacy,
        "name": f"{dataset}_cnn_spatial_seed42",
        "parameter_count": 400,
        "metrics": {"accuracy": 0.9, "macro_f1": 0.89},
    }
    baseline = {**record(dataset, 50), "name": f"{dataset}_linear_seed42"}
    for item in [legacy, spatial, baseline]:
        directory = tmp_path / item["name"]
        directory.mkdir()
        (directory / "test_result.json").write_text(json.dumps(item), encoding="utf-8")

    loaded = load_records(tmp_path, "test")
    assert {r["name"] for r in loaded} == {spatial["name"], baseline["name"]}
    # Direct callers must obey the same selection policy as the CLI loader.
    for inputs in [loaded, [legacy, spatial, baseline]]:
        rows = aggregate(inputs)
        cnn = next(row for row in rows if row["model"] == "cnn")
        assert len(rows) == 2
        assert cnn["runs"] == 1
        assert cnn["accuracy_mean"] == 0.9
        assert cnn["parameter_count"] == 400


def test_comparison_does_not_fall_back_to_legacy_cnn(tmp_path):
    legacy = {
        **record("fashion_mnist", 50),
        "model": "cnn",
        "name": "fashion_mnist_cnn_seed42",
    }
    directory = tmp_path / legacy["name"]
    directory.mkdir()
    (directory / "test_result.json").write_text(json.dumps(legacy), encoding="utf-8")
    assert load_records(tmp_path, "test") == []
    assert aggregate([legacy]) == []
