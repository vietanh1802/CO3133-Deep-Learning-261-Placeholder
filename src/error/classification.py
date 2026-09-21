"""Classification-specific error-analysis contracts."""

from dataclasses import dataclass

import torch

from src.error.base import ErrorAnalyzer, register_error_analyzer
from src.error.selection import select_bottom_k, select_top_k
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

    def __init__(self, num_examples: int = 5, num_confusions: int = 5) -> None:
        if num_examples < 0:
            raise ValueError("num_examples must be non-negative")
        if num_confusions < 0:
            raise ValueError("num_confusions must be non-negative")
        self.num_examples = num_examples
        self.num_confusions = num_confusions

    def analyze(self, evaluation: EvaluationResult) -> ClassificationErrorReport:
        """Convert classification evaluation evidence into an error report."""
        targets = evaluation.targets
        predictions = evaluation.predictions
        probabilities = evaluation.probabilities
        sample_indices = evaluation.sample_indices

        if targets.ndim != 1 or predictions.ndim != 1 or sample_indices.ndim != 1:
            raise ValueError("targets, predictions, and sample_indices must be one-dimensional")
        if targets.shape != predictions.shape or targets.shape != sample_indices.shape:
            raise ValueError("evaluation evidence must contain one entry per sample")
        if probabilities.ndim != 2 or probabilities.shape[0] != targets.shape[0]:
            raise ValueError("probabilities must have shape [samples, classes]")
        confusion = evaluation.metrics.confusion_matrix
        if probabilities.shape[1] != confusion.shape[0]:
            raise ValueError("probability and confusion-matrix class counts must match")
        if not torch.isfinite(probabilities).all():
            raise ValueError("probabilities must be finite")
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("probabilities must lie in [0, 1]")
        expected_sums = torch.ones(
            probabilities.shape[0], dtype=probabilities.dtype, device=probabilities.device
        )
        if not torch.allclose(probabilities.sum(dim=1), expected_sums, atol=1e-5):
            raise ValueError("probability rows must sum to one")

        support = confusion.sum(dim=1)
        per_class_accuracy = torch.where(
            support > 0,
            confusion.diagonal().to(torch.float64) / support,
            torch.zeros_like(support, dtype=torch.float64),
        )

        confidence = probabilities.gather(1, predictions[:, None]).squeeze(1)
        true_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
        margin = confidence - true_probability
        correct = predictions == targets
        errors = ~correct

        confident_indices = select_top_k(confidence, errors, self.num_examples)
        uncertain_indices = select_bottom_k(confidence, errors, self.num_examples)
        correct_indices = select_top_k(confidence, correct, self.num_examples)

        pair_counts: list[ConfusionPair] = []
        for target in range(confusion.shape[0]):
            for prediction in range(confusion.shape[1]):
                count = int(confusion[target, prediction])
                if target != prediction and count > 0:
                    pair_counts.append(ConfusionPair(target, prediction, count))
        pair_counts.sort(key=lambda pair: (-pair.count, pair.target, pair.prediction))

        def examples(indices: torch.Tensor) -> tuple[ClassificationError, ...]:
            return tuple(
                ClassificationError(
                    sample_index=int(sample_indices[index]),
                    target=int(targets[index]),
                    prediction=int(predictions[index]),
                    confidence=float(confidence[index]),
                    true_class_probability=float(true_probability[index]),
                    margin=float(margin[index]),
                )
                for index in indices.tolist()
            )

        return ClassificationErrorReport(
            per_class_accuracy=per_class_accuracy,
            per_class_support=support,
            most_confused_pairs=tuple(pair_counts[: self.num_confusions]),
            confident_errors=examples(confident_indices),
            uncertain_errors=examples(uncertain_indices),
            representative_correct=examples(correct_indices),
        )
