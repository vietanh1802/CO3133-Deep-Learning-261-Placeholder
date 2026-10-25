"""Hand-computed segmentation metrics and evidence command checks."""

import json
import subprocess
import sys

import pytest
import torch

from src.evaluation.segmentation import (
    boundary_mask,
    confusion_matrix,
    evaluate_predictions,
    metrics_from_confusion,
    select_examples,
)


def test_hand_computed_metrics():
    counts = confusion_matrix(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1, 1]), num_classes=3)
    assert counts.tolist() == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]
    result = metrics_from_confusion(counts)
    assert result.per_class_iou.tolist() == pytest.approx([0.5, 2 / 3, 0])
    assert result.per_class_dice.tolist() == pytest.approx([2 / 3, 0.8, 0])
    assert result.mean_iou == pytest.approx((0.5 + 2 / 3) / 3)


def test_regions_and_selection():
    targets = torch.tensor([[[0, 0, 1, 1, 1]], [[4, 4, 4, 4, 4]]])
    predictions = targets.clone()
    predictions[0] = 0
    assert boundary_mask(targets).tolist() == [
        [[False, True, True, False, False]],
        [[False, False, False, False, False]],
    ]
    result = evaluate_predictions(targets, predictions)
    assert torch.equal(
        result["all"].confusion_matrix,
        result["boundary"].confusion_matrix + result["interior"].confusion_matrix,
    )
    assert select_examples(targets, predictions) == {"best": 1, "worst": 0, "robot": 1}
    # Split metrics use pooled counts, not a mean of image scores.
    assert result["all"].per_class_iou[0] == pytest.approx(2 / 5)


@pytest.mark.parametrize(
    "prediction", [torch.tensor([0.0]), torch.tensor([-1]), torch.tensor([8]), torch.tensor([True])]
)
def test_invalid_labels(prediction):
    with pytest.raises(ValueError):
        confusion_matrix(torch.tensor([0]), prediction)


def test_empty_and_mismatched_masks():
    with pytest.raises(ValueError):
        evaluate_predictions(
            torch.empty((0, 2, 2), dtype=torch.long), torch.empty((0, 2, 2), dtype=torch.long)
        )
    with pytest.raises(ValueError):
        confusion_matrix(torch.zeros(2, dtype=torch.long), torch.zeros(3, dtype=torch.long))


def test_evidence_command(tmp_path):
    labels = torch.tensor([[[0, 4], [0, 4]]])
    evidence = tmp_path / "evidence.pt"
    output = tmp_path / "result"
    torch.save(
        {
            "targets": labels,
            "predictions": labels,
            "split": "validation",
            "images": torch.zeros(1, 3, 2, 2),
        },
        evidence,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluation.evaluate_segmentation",
            "--evidence",
            str(evidence),
            "--output",
            str(output),
        ],
        check=True,
    )
    record = json.loads((output / "metrics.json").read_text())
    assert record["metrics"]["all"]["mean_iou"] == pytest.approx(2 / 8)
    assert record["metrics"]["all"]["pixel_count"] == 4
    assert (output / "examples.png").is_file()
    assert (output / "confusion_matrix.png").is_file()


def test_model_batches_pool_counts_and_restore_mode():
    from src.evaluation.segmentation import evaluate_model

    class ToyModel(torch.nn.Module):
        def forward(self, images):
            return {"out": images}

    model = ToyModel()
    target = torch.tensor([[[0, 1, 1]]])
    logits = torch.tensor([[[[3.0, 3.0, 0.0]], [[0.0, 0.0, 3.0]]]])
    result = evaluate_model(
        model, [(logits, target), (logits, target)], torch.device("cpu"), num_classes=2
    )
    assert result["all"].confusion_matrix.tolist() == [[2, 0], [2, 2]]
    assert model.training
    with pytest.raises(ValueError):
        evaluate_model(model, [], torch.device("cpu"))
    assert model.training
