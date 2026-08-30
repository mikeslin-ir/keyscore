from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Protocol, Dict

import torch
import torch.nn.functional as F

# ---- copy the helper funcs from your module ----
import torch.nn.functional as F
def l2(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)
def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = l2(a); b = l2(b)
    return (a * b).sum(dim=-1)

def minmax(x: torch.Tensor) -> torch.Tensor:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8) if (mx - mn) > 1e-8 else torch.zeros_like(x)



# ---- Pooling protocol and simple poolers ----
class Pooling(Protocol):
    def __call__(self, frame_embs: torch.Tensor, *, text_emb: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ...

@dataclass
class MeanPooling(Pooling):
    def __call__(self, frame_embs: torch.Tensor, *, text_emb: Optional[torch.Tensor] = None):
        v = frame_embs.mean(dim=0)
        return l2(v), None

@dataclass
class TextGuidedAttentionPooling(Pooling):
    tau: float = 0.07  # temperature (lower = sharper)
    def __call__(self, frame_embs: torch.Tensor, *, text_emb: Optional[torch.Tensor] = None):
        if text_emb is None:
            raise ValueError("TextGuidedAttentionPooling requires text_emb.")
        # attention ~ softmax( cosine(frame_i, text) / tau )
        sims = cos(frame_embs, text_emb) / self.tau            # [N]
        attn = torch.softmax(sims, dim=0)                      # [N]
        v = (attn.unsqueeze(-1) * frame_embs).sum(dim=0)       # [D]
        return l2(v), attn


# -----------------------------
# FrameScorer
# -----------------------------
@dataclass
class FrameScorer:
    """
    Compute per-frame scores via three signals:
      1) Semantic similarity (text ↔ frame)
      2) Temporal representativeness (video ↔ frame)
      3) Frame-drop contribution (Δ cosine(text, video) when leaving the frame out)
    """
    pooling: Pooling
    alpha: float = 0.85   # semantic weight
    beta: float  = 0.05   # temporal weight
    gamma: float = 0.1   # drop-based contextual weight
    normalize_each: bool = True
    use_contextual: bool = True  # kept for compatibility

    def __post_init__(self):
        s = self.alpha + self.beta + self.gamma
        if abs(s - 1.0) > 1e-6:
            raise ValueError("alpha + beta + gamma must sum to 1.0")

    def compute(
        self,
        frame_embs: torch.Tensor,   # [N, D]
        text_emb: torch.Tensor,     # [D]
        video_emb: Optional[torch.Tensor] = None,   # if provided, bypass pooling for base video emb
    ) -> Dict[str, torch.Tensor]:
        if frame_embs.dim() != 2 or text_emb.dim() != 1:
            raise ValueError("Shapes must be frame_embs=[N,D], text_emb=[D].")

        f = l2(frame_embs)   # [N, D]  unit-norm frame embeddings
        t = l2(text_emb)     # [D]     unit-norm text embedding

        # Build or use provided video embedding
        attn = None
        if video_emb is None:
            v, attn = self.pooling(f, text_emb=t)
        else:
            v = l2(video_emb)

        # --- Components ---
        # All embeddings are already unit-norm; dot product == cosine similarity.
        # 1) Semantic: score(v_i, w) = v_i · w  (equation 1)
        sim_semantic = (f * t).sum(dim=-1)        # [N]

        # 2) Temporal Representation: cosine(frame_i, video)
        sim_temporal = (f * v).sum(dim=-1)        # [N]

        # 3) Frame-drop score: contribution to text–video similarity
        #    sim_drop[i] = cos(t, v_full) - cos(t, v_without_frame_i)
        base_sim = (v * t).sum(dim=-1)            # [] (scalar)
        N, D = f.shape

        if N <= 1:
            # nothing to drop
            sim_drop = torch.zeros(N, device=f.device, dtype=f.dtype)
        elif isinstance(self.pooling, MeanPooling):
            # Vectorized leave-one-out for mean pooling
            sum_f = f.sum(dim=0)                                  # [D]
            v_loo = (sum_f.unsqueeze(0) - f) / (N - 1)            # [N, D]
            v_loo = l2(v_loo)                                     # [N, D]
            # cos for each leave-one-out video vs text
            sim_loo = (v_loo * t.unsqueeze(0)).sum(dim=-1)        # [N]
            sim_drop = base_sim - sim_loo                         # [N]
        else:
            # Generic fallback: recompute pooling N times without frame i
            drops = []
            for i in range(N):
                mask = torch.ones(N, dtype=torch.bool, device=f.device)
                mask[i] = False
                v_minus_i, _ = self.pooling(f[mask], text_emb=t)  # [D]
                sim_minus_i = cos(v_minus_i, t)                   # []
                drops.append((base_sim - sim_minus_i).unsqueeze(0))
            sim_drop = torch.cat(drops, dim=0)                    # [N]

        # Normalize each component if requested
        if self.normalize_each:
            sim_semantic = minmax(sim_semantic)
            sim_temporal = minmax(sim_temporal)
            sim_drop     = minmax(sim_drop)

        scores = self.alpha * sim_semantic + self.beta * sim_temporal + self.gamma * sim_drop

        return {
            "scores": scores,          # [N]
            "sim_semantic": sim_semantic,      # [N]
            "sim_temporal": sim_temporal,    # [N]
            "sim_drop": sim_drop,      # [N]
            "video_emb": v,            # [D]
            "attn": attn,              # [N] or None
        }

# # ---- Usage Example ----
# from FrameScorer import FrameScorer

# # -------------- Example A: Mean pooling --------------
# N, D = 12, 256
# frame_embs = torch.randn(N, D)
# text_emb   = torch.randn(D)

# frame_scorer = FrameScorer(
#     pooling=MeanPooling(),
#     alpha=0.7,  # semantic
#     beta=0.1,   # temporal
#     gamma=0.2,  # drop
#     normalize_each=True
# )

# out_frame_scores = frame_scorer.compute(frame_embeddings, text_embedding)
# print("MeanPooling")
# print("scores shape:", out_frame_scores["scores"].shape)
# topk = torch.topk(out_scorer["scores"], k=5)
# print("top-5 frames (hybrid):", topk.indices.tolist())
# print("sim_semantic: ", out_frame_scores["sim_semantic"][:])
# print("sim_temporal:", out_frame_scores["sim_temporal"][:])
# print("sim_drop:", out_frame_scores["sim_drop"][:])
# print()

# # -------------- Example B: Text-guided attention pooling --------------
# scorer_attn = FrameScorer(
#     pooling=TextGuidedAttentionPooling(tau=0.07),
#     alpha=0.5,
#     beta=0.3,
#     gamma=0.2,
#     normalize_each=True
# )

# out_attn = scorer_attn.compute(frame_embs, text_emb)
# print("B) TextGuidedAttentionPooling")
# print("scores shape:", out_attn["scores"].shape)
# print("attn shape:", None if out_attn["attn"] is None else out_attn["attn"].shape)
# topk = torch.topk(out_attn["scores"], k=5)
# print("top-5 frames (hybrid):", topk.indices.tolist())

# # Access individual methods if you want the three separate score types:
# sim_semantic_only = out_attn["sim_text"]     # 1) semantic score per frame
# sim_temporal_only = out_attn["sim_video"]    # 2) temporal representativeness per frame
# sim_drop_only     = out_attn["sim_drop"]     # 3) frame-drop contribution per frame
# hybrid_scores     = out_attn["scores"]       # 4) hybrid weighted sum
