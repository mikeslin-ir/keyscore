"""Controlled query-aware frame-selection comparison on MSR-VTT retrieval.

All methods share the SAME frozen PE backbone and the SAME STACFP candidate
frames; they differ ONLY in how the final frames are selected. This isolates
the contribution of KeyScore's scoring from the effect of merely using the
query, directly addressing the "is Table 5 a fair comparison?" concern.

Methods compared (per-video frame-budget-matched):
  * STACFP-mean        : mean-pool ALL STACFP candidate frames (no selection).
  * UFP-8              : uniform 8-frame sampling (query-agnostic reference).
  * QueryTopK (CLIP)   : top-K frames by text-frame cosine similarity, where K
                         equals the number of frames KeyScore keeps for that
                         video (frame-matched, query-aware baseline).
  * KeyScore (S+T+D)   : full query-conditioned utility, threshold selection.

Reported: R@1/5/10 (T2V & V2T) and average selected frames (ASF).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(os.environ.get("KEYSCORE_DATA_ROOT", "data"))

from keyscore.encoders.perception_encoder import PerceptionEncoder  # noqa: E402
from keyscore.proposals.stacfp import FrameSampler_STACFP  # noqa: E402
from keyscore.proposals.uniform import FrameSampler_Uniform  # noqa: E402
from keyscore.scoring.frame_scorer import FrameScorer, MeanPooling  # noqa: E402

TOP_K = [1, 5, 10]


def l2norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + 1e-8)


def mean_pool(frame_embs: torch.Tensor) -> torch.Tensor:
    v = frame_embs.mean(dim=0, keepdim=True)
    return l2norm(v, dim=-1)


def compute_recall(sim_matrix: np.ndarray, gt: List[int], k_vals: List[int]) -> Dict[str, float]:
    recalls = {k: 0 for k in k_vals}
    for index, sims in enumerate(sim_matrix):
        sorted_indices = np.argsort(sims)[::-1]
        for k in k_vals:
            if gt[index] in sorted_indices[:k]:
                recalls[k] += 1
    return {str(k): 100.0 * recalls[k] / len(gt) for k in recalls}


def evaluate(video_embs: List[torch.Tensor], text_embs: List[torch.Tensor]) -> Dict[str, Dict[str, float]]:
    videos = l2norm(torch.stack([e.squeeze(0) if e.ndim == 2 else e for e in video_embs]), dim=1)
    texts = l2norm(torch.stack([e.squeeze(0) if e.ndim == 2 else e for e in text_embs]), dim=1)
    gt = list(range(len(video_embs)))
    sim_t2v = (texts @ videos.T).numpy()
    sim_v2t = (videos @ texts.T).numpy()
    return {
        "recall_t2v": compute_recall(sim_t2v, gt, TOP_K),
        "recall_v2t": compute_recall(sim_v2t, gt, TOP_K),
    }


def load_ids(path: Path, limit: int | None) -> List[str]:
    with path.open("r") as handle:
        ids = [line.strip() for line in handle if line.strip()]
    return ids[:limit] if limit else ids


def load_captions(path: Path) -> Dict[str, str]:
    with path.open("r") as handle:
        annotations = json.load(handle)["annotations"]
    captions: Dict[str, str] = {}
    for ann in annotations:
        vid = ann.get("video_id", ann.get("image_id"))
        captions.setdefault(vid, ann["caption"])
    return captions


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def summarize(name: str, asf: List[int], video_embs: List[torch.Tensor], text_embs: List[torch.Tensor]) -> Dict:
    return {
        "name": name,
        "num_videos": len(video_embs),
        "avg_selected_frames": float(np.mean(asf)),
        "retrieval": evaluate(video_embs, text_embs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled query-aware selection comparison on MSR-VTT.")
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT / "MSRVTT")
    parser.add_argument("--list", type=Path,
                        default=DATA_ROOT / "MSRVTT/structured-symlinks/test_list_miech.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "wacv_query_aware")
    args = parser.parse_args()

    video_dir = args.dataset_root / "videos" / "all"
    captions = load_captions(args.dataset_root / "annotation" / "MSR_VTT.json")
    video_ids = load_ids(args.list, args.limit)
    missing = [v for v in video_ids if not (video_dir / f"{v}.mp4").exists()]
    if missing:
        raise FileNotFoundError(f"Missing videos: {missing[:5]} ...")

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "_frames"
    reset_dir(work)

    encoder = PerceptionEncoder()
    sampler = FrameSampler_STACFP(frame_step=5, gamma_time=5, k_min=6, k_max=30)
    ufp = FrameSampler_Uniform(no_of_frames=8)
    scorer = FrameScorer(pooling=MeanPooling(), alpha=0.7, beta=0.1, gamma=0.2, normalize_each=True)

    # Accumulators: video embeddings and matched text embeddings per method.
    methods = ["stacfp_mean", "ufp8", "query_topk", "keyscore"]
    vid_embs: Dict[str, List[torch.Tensor]] = {m: [] for m in methods}
    txt_embs: Dict[str, List[torch.Tensor]] = {m: [] for m in methods}
    asf: Dict[str, List[int]] = {m: [] for m in methods}

    t_start = time.perf_counter()
    for i, vid in enumerate(video_ids):
        video_path = video_dir / f"{vid}.mp4"
        text_vec = encoder.compute_text_emb(captions[vid]).squeeze(0)

        # --- shared STACFP candidate frames + embeddings ---
        cand_dir = work / "stacfp" / vid
        sampler.extract_frames(str(video_path), str(cand_dir))
        cand_frames, _ = encoder.compute_video_emb(video_frame_folder=str(cand_dir), return_frame_embs=True)

        out = scorer.compute(cand_frames, text_vec)
        scores = out["scores"]
        sim_semantic = out["sim_semantic"]

        # KeyScore selection (threshold, >=1 frame).
        ks_mask = scores >= args.threshold
        if not bool(ks_mask.any()):
            ks_mask[int(torch.argmax(scores).item())] = True
        ks_idx = torch.nonzero(ks_mask, as_tuple=False).squeeze(-1)
        n_keep = int(ks_idx.numel())

        # Query-aware Top-K by text-frame similarity, frame-matched to KeyScore.
        k = min(n_keep, cand_frames.shape[0])
        qa_idx = torch.topk(sim_semantic, k=k).indices

        vid_embs["stacfp_mean"].append(mean_pool(cand_frames).squeeze(0))
        asf["stacfp_mean"].append(cand_frames.shape[0])

        vid_embs["query_topk"].append(mean_pool(cand_frames[qa_idx]).squeeze(0))
        asf["query_topk"].append(k)

        vid_embs["keyscore"].append(mean_pool(cand_frames[ks_idx]).squeeze(0))
        asf["keyscore"].append(n_keep)

        # --- UFP-8 reference (query-agnostic) ---
        ufp_dir = work / "ufp" / vid
        ufp.extract_frames(str(video_path), str(ufp_dir))
        ufp_frames, _ = encoder.compute_video_emb(video_frame_folder=str(ufp_dir), return_frame_embs=True)
        vid_embs["ufp8"].append(mean_pool(ufp_frames).squeeze(0))
        asf["ufp8"].append(ufp_frames.shape[0])

        for m in methods:
            txt_embs[m].append(text_vec)

        # free per-video frame dirs to limit disk usage
        shutil.rmtree(cand_dir, ignore_errors=True)
        shutil.rmtree(ufp_dir, ignore_errors=True)

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"[{i + 1}/{len(video_ids)}] elapsed={elapsed:.1f}s", flush=True)

    results = {
        "metadata": {
            "list": str(args.list),
            "num_videos": len(video_ids),
            "threshold": args.threshold,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "scorer": {"alpha": 0.7, "beta": 0.1, "gamma": 0.2},
            "stacfp": {"frame_step": 5, "gamma_time": 5, "k_min": 6, "k_max": 30},
            "note": "All methods share PE backbone + STACFP candidates; QueryTopK is frame-matched to KeyScore.",
        }
    }
    for m in methods:
        results[m] = summarize(m, asf[m], vid_embs[m], txt_embs[m])

    out_json = args.out / "msrvtt_query_aware.json"
    with out_json.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps({m: {"R@1_t2v": results[m]["retrieval"]["recall_t2v"]["1"],
                          "R@1_v2t": results[m]["retrieval"]["recall_v2t"]["1"],
                          "ASF": results[m]["avg_selected_frames"]} for m in methods}, indent=2))
    print(f"Saved: {out_json}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
