from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar

EvaluationT = TypeVar("EvaluationT")
ReportT = TypeVar("ReportT")


class ErrorAnalyzer(ABC, Generic[EvaluationT, ReportT]):
    """Convert task-specific evaluation evidence into an error report."""

    @abstractmethod
    def analyze(self, evaluation: EvaluationT) -> ReportT:
        """Convert evaluation evidence into a structured error report."""
        raise NotImplementedError


Analyzer = ErrorAnalyzer[Any, Any]
AnalyzerT = TypeVar("AnalyzerT", bound=type[Analyzer])
_ANALYZER_REGISTRY: dict[str, type[Analyzer]] = {}


def register_error_analyzer(name: str) -> Callable[[AnalyzerT], AnalyzerT]:
    """Register an analyzer class under a normalized task name."""
    key = name.strip().lower().replace("-", "_")
    if not key:
        raise ValueError("Analyzer name cannot be empty")

    def decorator(analyzer_class: AnalyzerT) -> AnalyzerT:
        if key in _ANALYZER_REGISTRY:
            raise ValueError(f"Error analyzer {key!r} is already registered")
        _ANALYZER_REGISTRY[key] = analyzer_class
        return analyzer_class

    return decorator


def build_error_analyzer(
    name: str,
    config: Mapping[str, Any] | None = None,
) -> Analyzer:
    """Construct one registered error analyzer."""
    key = name.strip().lower().replace("-", "_")
    if key not in _ANALYZER_REGISTRY:
        available = ", ".join(sorted(_ANALYZER_REGISTRY)) or "none"
        raise ValueError(f"Unknown error analyzer {name!r}. Available analyzers: {available}")
    return _ANALYZER_REGISTRY[key](**dict(config or {}))
