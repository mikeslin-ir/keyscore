"""STACFP temporal-weight (gamma_time) sensitivity sweep on MSR-VTT retrieval.

For each gamma_time value we re-run STACFP proposal + PE embedding + KeyScore
selection and report:
  * avg_clusters  : mean silhouette-chosen cluster count (chosen_k) of STACFP.
  * avg_proposals : mean number of candidate frames produced by STACFP.
  * avg_selected  : mean number of frames kept by KeyScore (ASF).
  * R@1 (T2V/V2T) : zero-shot retrieval accuracy.

This isolates the effect of the temporal weight gamma_time in Eq. (3),
as requested by the reviewer (distinct from the alpha/beta/gamma scoring sweep
and from the binary timestamp-normalization ablation).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
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
from keyscore.scoring.frame_scorer import FrameScorer, MeanPooling  # noqa: E402

TOP_K = [1, 5, 10]


def l2norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + 1e-8)


def mean_pool(frame_embs: torch.Tensor) -> torch.Tensor:
    return l2norm(frame_embs.mean(dim=0, keepdim=True), dim=-1)


def compute_recall(sim_matrix: np.ndarray, gt: List[int], k_vals: List[int]) -> Dict[str, float]:
    recalls = {k: 0 for k in k_vals}
    for index, sims in enumerate(sim_matrix):
        order = np.argsort(sims)[::-1]
        for k in k_vals:
            if gt[index] in order[:k]:
                recalls[k] += 1
    return {str(k): 100.0 * recalls[k] / len(gt) for k in recalls}


def evaluate(video_embs: List[torch.Tensor], text_embs: List[torch.Tensor]) -> Dict[str, Dict[str, float]]:
    videos = l2norm(torch.stack(video_embs), dim=1)
    texts = l2norm(torch.stack(text_embs), dim=1)
    gt = list(range(len(video_embs)))
    return {
        "recall_t2v": compute_recall((texts @ videos.T).numpy(), gt, TOP_K),
        "recall_v2t": compute_recall((videos @ texts.T).numpy(), gt, TOP_K),
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


def read_chosen_k(frame_dir: Path, fallback: int) -> int:
    meta = frame_dir / "stpac_meta.json"
    if meta.exists():
        try:
            with meta.open("r") as handle:
                data = json.load(handle)
            return int(data.get("chosen_k", fallback))
        except Exception:
            return fallback
    return fallback


def run_gamma(
    *,
    gamma_time: float,
    encoder: PerceptionEncoder,
    scorer: FrameScorer,
    video_ids: List[str],
    captions: Dict[str, str],
    video_dir: Path,
    work: Path,
    threshold: float,
) -> Dict:
    sampler = FrameSampler_STACFP(frame_step=5, gamma_time=gamma_time, k_min=6, k_max=30)
    stage = work / f"gamma_{gamma_time}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    video_embs: List[torch.Tensor] = []
    text_embs: List[torch.Tensor] = []
    clusters: List[int] = []
    proposals: List[int] = []
    selected: List[int] = []

    for i, vid in enumerate(video_ids):
        video_path = video_dir / f"{vid}.mp4"
        frame_dir = stage / vid
        sampler.extract_frames(str(video_path), str(frame_dir))
        frame_embs, _ = encoder.compute_video_emb(video_frame_folder=str(frame_dir), return_frame_embs=True)
        text_vec = encoder.compute_text_emb(captions[vid]).squeeze(0)

        n_prop = int(frame_embs.shape[0])
        clusters.append(read_chosen_k(frame_dir, fallback=n_prop))
        proposals.append(n_prop)

        scores = scorer.compute(frame_embs, text_vec)["scores"]
        mask = scores >= threshold
        if not bool(mask.any()):
            mask[int(torch.argmax(scores).item())] = True
        sel = frame_embs[mask]
        selected.append(int(sel.shape[0]))

        video_embs.append(mean_pool(sel).squeeze(0))
        text_embs.append(text_vec)

        shutil.rmtree(frame_dir, ignore_errors=True)
        if (i + 1) % 50 == 0:
            print(f"  gamma={gamma_time} [{i + 1}/{len(video_ids)}]", flush=True)

    shutil.rmtree(stage, ignore_errors=True)
    return {
        "gamma_time": gamma_time,
        "num_videos": len(video_ids),
        "avg_clusters": float(np.mean(clusters)),
        "avg_proposals": float(np.mean(proposals)),
        "avg_selected_frames": float(np.mean(selected)),
        "retrieval": evaluate(video_embs, text_embs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="STACFP gamma_time sensitivity sweep on MSR-VTT.")
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT / "MSRVTT")
    parser.add_argument("--list", type=Path,
                        default=DATA_ROOT / "MSRVTT/structured-symlinks/test_list_miech.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "wacv_gamma_sweep")
    args = parser.parse_args()

    video_dir = args.dataset_root / "videos" / "all"
    captions = load_captions(args.dataset_root / "annotation" / "MSR_VTT.json")
    video_ids = load_ids(args.list, args.limit)
    missing = [v for v in video_ids if not (video_dir / f"{v}.mp4").exists()]
    if missing:
        raise FileNotFoundError(f"Missing videos: {missing[:5]} ...")

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "_frames"
    work.mkdir(parents=True, exist_ok=True)

    encoder = PerceptionEncoder()
    scorer = FrameScorer(pooling=MeanPooling(), alpha=0.7, beta=0.1, gamma=0.2, normalize_each=True)

    results = {
        "metadata": {
            "list": str(args.list),
            "num_videos": len(video_ids),
            "threshold": args.threshold,
            "gammas": args.gammas,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "note": "Sensitivity of STACFP temporal weight gamma_time (Eq. 3).",
        },
        "sweep": [],
    }
    for g in args.gammas:
        print(f"=== gamma_time={g} ===", flush=True)
        res = run_gamma(
            gamma_time=g,
            encoder=encoder,
            scorer=scorer,
            video_ids=video_ids,
            captions=captions,
            video_dir=video_dir,
            work=work,
            threshold=args.threshold,
        )
        results["sweep"].append(res)
        print(json.dumps({"gamma": g,
                          "avg_clusters": round(res["avg_clusters"], 2),
                          "avg_selected": round(res["avg_selected_frames"], 2),
                          "R@1_t2v": res["retrieval"]["recall_t2v"]["1"],
                          "R@1_v2t": res["retrieval"]["recall_v2t"]["1"]}, indent=2), flush=True)

    out_json = args.out / "msrvtt_gamma_sweep.json"
    with out_json.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved: {out_json}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
