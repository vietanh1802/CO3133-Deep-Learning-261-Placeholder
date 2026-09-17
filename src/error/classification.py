"""Classification-specific error-analysis contracts."""

from dataclasses import dataclass

import torch

from src.error.base import ErrorAnalyzer, register_error_analyzer
from src.evaluation import EvaluationResult


@dataclass(frozen=True)
class ClassificationError:
    """One classified sample retained for qualitative analysis."""

    sample_index: int
    target: int
    prediction: int
    confidence: float
    true_class_probability: float
    margin: float


@dataclass(frozen=True)
class ConfusionPair:
    """One directed true-class to predicted-class confusion."""

    target: int
    prediction: int
    count: int


@dataclass(frozen=True)
class ClassificationErrorReport:
    """Structured evidence produced by classification error analysis."""

    per_class_accuracy: torch.Tensor
    per_class_support: torch.Tensor
    most_confused_pairs: tuple[ConfusionPair, ...]
    confident_errors: tuple[ClassificationError, ...]
    uncertain_errors: tuple[ClassificationError, ...]
    representative_correct: tuple[ClassificationError, ...]


@register_error_analyzer("classification")
class ClassificationErrorAnalyzer(ErrorAnalyzer[EvaluationResult, ClassificationErrorReport]):
    """Analyze classification predictions and confidence patterns."""

    def analyze(self, evaluation: EvaluationResult) -> ClassificationErrorReport:
        """Convert classification evaluation evidence into an error report."""
        raise NotImplementedError
