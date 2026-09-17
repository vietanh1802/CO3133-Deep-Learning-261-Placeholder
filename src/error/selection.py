"""Task-independent helpers for selecting ranked evaluation examples."""

import torch


def select_top_k(scores: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Return source indices of the highest masked scores."""
    return _select_k(scores, mask, k, descending=True)


def select_bottom_k(scores: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Return source indices of the lowest masked scores."""
    return _select_k(scores, mask, k, descending=False)


def _select_k(
    scores: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    *,
    descending: bool,
) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if scores.shape != mask.shape:
        raise ValueError("scores and mask must have the same shape")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have boolean dtype")
    if scores.device != mask.device:
        raise ValueError("scores and mask must be on the same device")
    if k < 0:
        raise ValueError("k must be non-negative")

    candidate_indices = torch.nonzero(mask, as_tuple=False).flatten()
    if k == 0 or candidate_indices.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    candidate_scores = scores[candidate_indices]
    order = torch.argsort(candidate_scores, descending=descending, stable=True)
    return candidate_indices[order[:k]]
