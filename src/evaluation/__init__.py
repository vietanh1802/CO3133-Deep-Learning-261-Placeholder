"""Evaluation interfaces."""

from src.evaluation.eval import EvaluationResult, evaluate_model
from src.evaluation.metrics import ClassificationMetrics

__all__ = ["ClassificationMetrics", "EvaluationResult", "evaluate_model"]
