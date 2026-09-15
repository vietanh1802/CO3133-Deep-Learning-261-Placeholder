"""Prediction-error analysis contracts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorExample:
    """One misclassified example retained for qualitative analysis."""

    sample_index: int
    target: int
    prediction: int
    confidence: float
