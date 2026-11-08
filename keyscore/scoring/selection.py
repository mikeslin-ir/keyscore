from typing import Literal

import torch
import torch.nn.functional as F

from keyscore.scoring.frame_scorer import FrameScorer, MeanPooling

SelectionMethod = Literal["semantic", "semantic_drop", "keyscore", "greedy", "mmr"]
METHODS: tuple[SelectionMethod, ...] = (
    "semantic", "semantic_drop", "keyscore", "greedy", "mmr"
)
WEIGHTS = {
    "semantic": (1.0, 0.0, 0.0),
    "semantic_drop": (0.8, 0.0, 0.2),
    "keyscore": (0.7, 0.1, 0.2),
}


def _validate_embeddings(frame_embs: torch.Tensor, text_embs: torch.Tensor) -> None:
    if frame_embs.ndim != 2 or text_embs.ndim != 2:
        raise ValueError("Expected frame_embs=[N,D] and text_embs=[Q,D].")
    if min(frame_embs.shape) == 0 or min(text_embs.shape) == 0:
        raise ValueError("Embedding matrices must be nonempty.")
    if frame_embs.shape[1] != text_embs.shape[1]:
        raise ValueError("Frame and query embedding dimensions must match.")
    if frame_embs.device != text_embs.device or frame_embs.dtype != text_embs.dtype:
        raise ValueError("Frame and query embeddings must share a device and dtype.")
    if not frame_embs.is_floating_point() or not text_embs.is_floating_point():
        raise ValueError("Embeddings must be floating point.")
    if not bool(torch.isfinite(frame_embs).all() and torch.isfinite(text_embs).all()):
        raise ValueError("Embeddings must be finite.")
    if bool((frame_embs.norm(dim=1) < 1e-8).any() or (text_embs.norm(dim=1) < 1e-8).any()):
        raise ValueError("Zero-norm embeddings cannot be selected.")


def select_indices(
    frame_embs: torch.Tensor,
    text_embs: torch.Tensor,
    k: int,
    method: SelectionMethod,
    *,
    mmr_relevance: float = 0.7,
    keyscore_weights: tuple[float, float, float] = WEIGHTS["keyscore"],
) -> torch.Tensor:
    """Return [Q,min(K,N)] indices, with each query selecting independently."""
    _validate_embeddings(frame_embs, text_embs)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if method not in METHODS:
        raise ValueError(f"Unknown selector: {method}")
    if not 0.0 <= mmr_relevance <= 1.0:
        raise ValueError("mmr_relevance must be in [0,1].")

    f, t = F.normalize(frame_embs, dim=1), F.normalize(text_embs, dim=1)
    k = min(k, f.shape[0])
    if method in WEIGHTS:
        alpha, beta, gamma = keyscore_weights if method == "keyscore" else WEIGHTS[method]
        scorer = FrameScorer(MeanPooling(), alpha=alpha, beta=beta, gamma=gamma)
        scores = scorer.compute_queries(f, t)["scores"]
        return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]

    semantic = t @ f.T
    chosen = torch.zeros_like(semantic, dtype=torch.bool)
    rows = torch.arange(t.shape[0], device=t.device)
    indices = torch.empty((t.shape[0], k), dtype=torch.long, device=t.device)
    sums = torch.zeros_like(t)
    gram = f @ f.T
    redundancy = torch.zeros_like(semantic)
    for step in range(k):
        if method == "greedy":
            numerator = (sums * t).sum(dim=1, keepdim=True) + semantic
            squared_norm = sums.square().sum(dim=1, keepdim=True) + 1.0 + 2.0 * sums @ f.T
            scores = numerator / squared_norm.clamp_min(1e-12).sqrt()
        else:
            scores = mmr_relevance * semantic - (1.0 - mmr_relevance) * redundancy
        scores = scores.masked_fill(chosen, -torch.inf)
        index = scores.argmax(dim=1)
        indices[:, step] = index
        chosen[rows, index] = True
        sums = sums + f[index]
        if method == "mmr":
            similarities = gram[index]
            redundancy = similarities if step == 0 else torch.maximum(redundancy, similarities)
    return indices


def pooled_pairs(
    frame_embs: torch.Tensor, text_embs: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pair scores and selected means; callers supply validated indices."""
    selected = F.normalize(frame_embs, dim=1)[indices].mean(dim=1)
    if bool((selected.norm(dim=1) < 1e-8).any()):
        raise ValueError("Selected frames have a zero-norm mean; cosine alignment is undefined.")
    pooled = F.normalize(selected, dim=1)
    return (pooled * F.normalize(text_embs, dim=1)).sum(dim=1), pooled
