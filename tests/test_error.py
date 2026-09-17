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
