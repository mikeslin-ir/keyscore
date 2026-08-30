"""
Top-K frame selector using FrameScorer.

Given pre-computed frame embeddings and a text query (question),
scores every candidate frame and returns the top-K paths in temporal order.

Two selection modes:
  - 'scored'  : top-K by FrameScorer composite score  (caption-guided)
  - 'uniform' : uniformly spaced indices               (blind baseline)
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Literal

import torch

# Resolve project src/ so sibling packages are importable
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from keyscore.scoring.frame_scorer import FrameScorer, MeanPooling
from keyscore.encoders.clip_encoder import ClipEncoder


class FrameSelector:
    """
    Loads pre-computed frame embeddings produced by embed_frames.py
    and selects top-K frames per question.
    """

    def __init__(
        self,
        encoder: ClipEncoder | None = None,
        alpha: float = 0.85,
        beta: float = 0.05,
        gamma: float = 0.10,
    ):
        # amp=False: text-only encoding doesn't benefit from AMP and triggers
        # a BFloat16 overflow in CLIP's causal attention mask on some GPUs
        self.encoder = encoder or ClipEncoder(batch_size=512, amp=False, num_workers=0)
        self.scorer = FrameScorer(MeanPooling(), alpha=alpha, beta=beta, gamma=gamma)

    # ------------------------------------------------------------------
    def select_from_arrays(
        self,
        frame_embs: torch.Tensor,
        frame_paths: list[str],
        question: str,
        k: int,
        mode: Literal["scored", "uniform"] = "scored",
    ) -> tuple[list[str], list[float]]:
        """
        Select top-K frame paths using pre-loaded embeddings (no disk I/O).
        Call this in a loop over questions after loading frame_embs/frame_paths once per video.
        """
        N = frame_embs.shape[0]
        k_eff = min(k, N)

        if mode == "uniform":
            indices = _uniform_indices(N, k_eff)
            scores_list = [0.0] * k_eff
        else:
            text_emb = self.encoder.embed_text(question)
            result = self.scorer.compute(frame_embs, text_emb)
            scores = result["scores"]
            topk = scores.topk(k_eff)
            indices = sorted(topk.indices.tolist())
            scores_list = scores[indices].tolist()

        selected_paths = [frame_paths[i] for i in indices]
        return selected_paths, scores_list

    def select_from_arrays_with_text_emb(
        self,
        frame_embs: torch.Tensor,
        frame_paths: list[str],
        text_emb: torch.Tensor,
        k: int,
    ) -> tuple[list[str], list[float]]:
        """
        Scored-mode only: select top-K paths using pre-computed text embedding (no encoder call).
        Use after batch embed_text(questions) to avoid one CLIP call per question.
        """
        if text_emb.dim() == 2:
            text_emb = text_emb.squeeze(0)
        N = frame_embs.shape[0]
        k_eff = min(k, N)
        result = self.scorer.compute(frame_embs, text_emb)
        scores = result["scores"]
        topk = scores.topk(k_eff)
        indices = sorted(topk.indices.tolist())
        scores_list = scores[indices].tolist()
        selected_paths = [frame_paths[i] for i in indices]
        return selected_paths, scores_list

    def select(
        self,
        emb_dir: str | Path,
        question: str,
        k: int,
        mode: Literal["scored", "uniform"] = "scored",
    ) -> tuple[list[str], list[float]]:
        """
        Load embeddings from disk and select top-K. For many questions per video,
        prefer loading once and calling select_from_arrays() in a loop.
        """
        emb_dir = Path(emb_dir)
        frame_embs = torch.load(emb_dir / "frame_embeddings.pt", weights_only=True)
        frame_paths = json.loads((emb_dir / "frame_paths.json").read_text())
        return self.select_from_arrays(frame_embs, frame_paths, question, k, mode)


# ------------------------------------------------------------------
def _uniform_indices(total: int, k: int) -> list[int]:
    """Evenly-spaced indices, inclusive of first and last frame."""
    if k == 1:
        return [total // 2]
    step = (total - 1) / (k - 1)
    return [round(i * step) for i in range(k)]


def get_uniform_frame_paths(frame_paths: list[str], k: int) -> list[str]:
    """Return K uniformly spaced frame paths (no encoder, no disk). Use for uniform mode."""
    N = len(frame_paths)
    k_eff = min(k, N)
    indices = _uniform_indices(N, k_eff)
    return [frame_paths[i] for i in indices]
