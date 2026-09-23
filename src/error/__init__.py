"""Prediction-error analysis contracts and construction helpers."""

from src.error.base import ErrorAnalyzer, build_error_analyzer, register_error_analyzer
from src.error.classification import (
    ClassificationError,
    ClassificationErrorAnalyzer,
    ClassificationErrorReport,
    ConfusionPair,
)
from src.error.selection import select_bottom_k, select_top_k

__all__ = [
    "ClassificationError",
    "ClassificationErrorAnalyzer",
    "ClassificationErrorReport",
    "ConfusionPair",
    "ErrorAnalyzer",
    "build_error_analyzer",
    "register_error_analyzer",
    "select_bottom_k",
    "select_top_k",
]
