"""Small, centralized experiment-logging helpers."""

import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from typing import Any

from loguru import logger


class ExperimentState(StrEnum):
    """Shared vocabulary for experiment lifecycle messages."""

    INITIALIZING = "Initializing"
    LOADING_DATA = "Loading data"
    TRAINING = "Training"
    VALIDATING = "Validating"
    TESTING = "Testing"
    SAVING = "Saving results"
    COMPLETED = "Completed"
    FAILED = "Failed"


def setup_logging(level: str = "INFO") -> None:
    """Configure one concise Loguru console handler."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def log_config(config: Mapping[str, Any]) -> None:
    """Log a resolved experiment configuration."""
    logger.info("Experiment configuration")
    for key, value in config.items():
        logger.info("  {}: {}", key, value)


@contextmanager
def log_timing(operation: str) -> Iterator[None]:
    """Log an operation's duration and re-raise any failure."""
    start = time.perf_counter()
    logger.info("Starting: {}", operation)
    try:
        yield
    except Exception:
        logger.exception("Failed: {} after {:.2f}s", operation, time.perf_counter() - start)
        raise
    logger.success("Completed: {} ({:.2f}s)", operation, time.perf_counter() - start)
