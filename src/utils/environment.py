"""Software and source-revision records that make reported results traceable."""

import platform
import subprocess
from typing import Any

import torch
import torchvision


def _git(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def software_versions() -> dict[str, Any]:
    """Return the Python, PyTorch, torchvision, CUDA, and cuDNN versions in use."""
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda if cuda_available else None,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
    }


def git_revision() -> dict[str, Any]:
    """Return the current commit and whether tracked or untracked files differ from it."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}
