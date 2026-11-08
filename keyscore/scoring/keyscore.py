"""Public, mean-pooling API for the controlled manuscript configuration."""

from dataclasses import dataclass
import math
from typing import TypedDict

import torch

from .frame_scorer import FrameScorer, MeanPooling
from .selection import (
    SelectionMethod,
    _validate_embeddings,
    pooled_pairs,
    select_indices,
)


class FrameScores(TypedDict):
    """Per-candidate tensors: [N] for one query, [Q,N] for multiple queries."""

    scores: torch.Tensor
    semantic: torch.Tensor
    representativeness: torch.Tensor
    drop_impact: torch.Tensor


@dataclass(frozen=True)
class SelectionResult:
    """Indices are ranked by selection order, not by video timestamp.

    With one query, shapes are [min(K,N)], [D], and [].
    With multiple queries, shapes are [Q,min(K,N)], [Q,D], and [Q].
    """

    indices: torch.Tensor
    pooled_embeddings: torch.Tensor
    similarities: torch.Tensor


class KeyScore:
    """Training-free frame selection from shared-space frame/text embeddings.

    The public API uses mean pooling and per-query, per-component min--max
    normalization. Defaults match the current manuscript, not the historical
    FrameScorer defaults. No encoder or model weights are loaded by this class.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.1,
        gamma: float = 0.2,
        *,
        mmr_relevance: float = 0.7,
    ) -> None:
        self._scorer = FrameScorer(MeanPooling(), alpha, beta, gamma)
        if not math.isfinite(mmr_relevance) or not 0.0 <= mmr_relevance <= 1.0:
            raise ValueError("mmr_relevance must be finite and in [0,1].")
        self.mmr_relevance = mmr_relevance

    @property
    def weights(self) -> tuple[float, float, float]:
        return self._scorer.alpha, self._scorer.beta, self._scorer.gamma

    @staticmethod
    def _queries(text_embs: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if text_embs.ndim not in (1, 2):
            raise ValueError("Expected text_embs=[D] or [Q,D].")
        single = text_embs.ndim == 1
        return (text_embs.unsqueeze(0) if single else text_embs), single

    def score(self, frame_embs: torch.Tensor, text_embs: torch.Tensor) -> FrameScores:
        """Return the weighted score and its three normalized components."""
        queries, single = self._queries(text_embs)
        _validate_embeddings(frame_embs, queries)
        values = self._scorer.compute_queries(frame_embs, queries)
        if single:
            values = {name: value[0] for name, value in values.items()}
        return {
            "scores": values["scores"],
            "semantic": values["sim_semantic"],
            "representativeness": values["sim_temporal"],
            "drop_impact": values["sim_drop"],
        }

    def select(
        self,
        frame_embs: torch.Tensor,
        text_embs: torch.Tensor,
        k: int = 4,
        *,
        method: SelectionMethod = "keyscore",
    ) -> SelectionResult:
        """Select independently per query and return normalized means/cosines.

        Methods are keyscore, semantic, semantic_drop, greedy, and mmr.
        Custom weights affect keyscore only; the two score-based baselines
        retain their prespecified manuscript weights. K is capped at N.
        """
        queries, single = self._queries(text_embs)
        indices = select_indices(
            frame_embs,
            queries,
            k,
            method,
            mmr_relevance=self.mmr_relevance,
            keyscore_weights=self.weights,
        )
        similarities, embeddings = pooled_pairs(frame_embs, queries, indices)
        if single:
            return SelectionResult(indices[0], embeddings[0], similarities[0])
        return SelectionResult(indices, embeddings, similarities)
