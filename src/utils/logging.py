"""Small, centralized experiment-logging helpers."""

import sys
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger
from tqdm.auto import tqdm

T = TypeVar("T")


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


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure concise console logging and an optional run log file."""
    logger.remove()
    logger.add(
        _tqdm_sink,
        level=level.upper(),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            path,
            level=level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        )


def _tqdm_sink(message: Any) -> None:
    """Write a Loguru message without corrupting an active progress bar."""
    tqdm.write(str(message), file=sys.stderr, end="")


def progress_bar(iterable: Iterable[T], **kwargs: Any) -> Iterable[T]:
    """Create a stderr progress bar compatible with the Loguru console sink."""
    return tqdm(iterable, file=sys.stderr, dynamic_ncols=True, **kwargs)


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
