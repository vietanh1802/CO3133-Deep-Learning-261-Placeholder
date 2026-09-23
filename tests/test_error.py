import pytest
import torch

from src.error import (
    ClassificationErrorAnalyzer,
    ErrorAnalyzer,
    build_error_analyzer,
    register_error_analyzer,
    select_bottom_k,
    select_top_k,
)
from src.evaluation import ClassificationMetrics, EvaluationResult


@register_error_analyzer("test-task")
class TestAnalyzer(ErrorAnalyzer[int, str]):
    __test__ = False

    def __init__(self, prefix: str = "result") -> None:
        self.prefix = prefix

    def analyze(self, evaluation: int) -> str:
        return f"{self.prefix}:{evaluation}"


def test_error_analyzer_factory_dispatches_and_passes_config() -> None:
    analyzer = build_error_analyzer("test_task", {"prefix": "value"})

    assert analyzer.analyze(3) == "value:3"


def test_error_analyzer_factory_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown error analyzer"):
        build_error_analyzer("unknown")


def test_error_analyzer_registry_rejects_duplicate_name() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @register_error_analyzer("test_task")
        class DuplicateAnalyzer(ErrorAnalyzer[int, str]):
            def analyze(self, evaluation: int) -> str:
                return str(evaluation)


def test_classification_error_analyzer_is_registered() -> None:
    analyzer = build_error_analyzer("classification")

    assert isinstance(analyzer, ClassificationErrorAnalyzer)


def test_classification_error_analyzer_builds_required_evidence() -> None:
    targets = torch.tensor([0, 0, 1, 1, 2])
    predictions = torch.tensor([0, 1, 1, 2, 1])
    probabilities = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.6, 0.2],
            [0.2, 0.25, 0.55],
            [0.1, 0.8, 0.1],
        ]
    )
    evaluation = EvaluationResult(
        metrics=ClassificationMetrics(
            accuracy=0.4,
            macro_f1=0.3,
            confusion_matrix=torch.tensor([[1, 1, 0], [0, 1, 1], [0, 1, 0]]),
        ),
        sample_indices=torch.tensor([10, 11, 12, 13, 14]),
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        parameter_count=10,
        inference_seconds=0.1,
    )

    report = ClassificationErrorAnalyzer(num_examples=2, num_confusions=2).analyze(evaluation)

    assert torch.allclose(
        report.per_class_accuracy, torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    )
    assert report.per_class_support.tolist() == [2, 2, 1]
    assert [(pair.target, pair.prediction) for pair in report.most_confused_pairs] == [
        (0, 1),
        (1, 2),
    ]
    assert [example.sample_index for example in report.confident_errors] == [14, 11]
    assert [example.sample_index for example in report.uncertain_errors] == [13, 11]
    assert [example.sample_index for example in report.representative_correct] == [10, 12]


def test_classification_error_analyzer_rejects_invalid_probabilities() -> None:
    evaluation = EvaluationResult(
        metrics=ClassificationMetrics(
            accuracy=1.0,
            macro_f1=1.0,
            confusion_matrix=torch.tensor([[1, 0], [0, 0]]),
        ),
        sample_indices=torch.tensor([0]),
        targets=torch.tensor([0]),
        predictions=torch.tensor([0]),
        probabilities=torch.tensor([[0.8, 0.1]]),
        parameter_count=1,
        inference_seconds=0.1,
    )

    with pytest.raises(ValueError, match="sum to one"):
        ClassificationErrorAnalyzer().analyze(evaluation)


def test_select_top_k_returns_ranked_source_indices() -> None:
    scores = torch.tensor([0.4, 0.9, 0.7, 0.9])
    mask = torch.tensor([True, False, True, True])

    selected = select_top_k(scores, mask, k=2)

    assert selected.tolist() == [3, 2]


def test_select_bottom_k_returns_all_available_candidates() -> None:
    scores = torch.tensor([0.4, 0.9, 0.1, 0.7])
    mask = torch.tensor([True, False, True, False])

    selected = select_bottom_k(scores, mask, k=5)

    assert selected.tolist() == [2, 0]


@pytest.mark.parametrize("k", [0, 3])
def test_select_k_returns_empty_long_tensor_without_candidates(k: int) -> None:
    scores = torch.tensor([0.4, 0.9])
    mask = torch.tensor([False, False])

    selected = select_top_k(scores, mask, k)

    assert selected.dtype == torch.long
    assert selected.shape == (0,)
    assert selected.device == scores.device


def test_select_k_validates_inputs() -> None:
    scores = torch.tensor([0.4, 0.9])

    with pytest.raises(TypeError, match="boolean"):
        select_top_k(scores, torch.tensor([1, 0]), k=1)
    with pytest.raises(ValueError, match="same shape"):
        select_top_k(scores, torch.tensor([True]), k=1)
    with pytest.raises(ValueError, match="non-negative"):
        select_top_k(scores, torch.tensor([True, True]), k=-1)
