from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

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


def sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def timed(fn: Callable):
    start = sync_time()
    result = fn()
    return result, sync_time() - start


def mean_pool(frame_embs: torch.Tensor) -> torch.Tensor:
    video_emb = frame_embs.mean(dim=0, keepdim=True)
    return video_emb / (video_emb.norm(dim=-1, keepdim=True) + 1e-8)


def compute_recall(sim_matrix: np.ndarray, gt: List[int], k_vals: List[int]) -> Dict[int, float]:
    recalls = {k: 0 for k in k_vals}
    for index, sims in enumerate(sim_matrix):
        sorted_indices = np.argsort(sims)[::-1]
        for k in k_vals:
            if gt[index] in sorted_indices[:k]:
                recalls[k] += 1
    return {k: 100.0 * recalls[k] / len(gt) for k in recalls}


def load_ids(path: Path, limit: int | None) -> List[str]:
    with path.open("r") as handle:
        video_ids = [line.strip() for line in handle if line.strip()]
    return video_ids[:limit] if limit else video_ids


def load_captions(path: Path) -> Dict[str, str]:
    with path.open("r") as handle:
        annotations = json.load(handle)["annotations"]
    captions: Dict[str, str] = {}
    for ann in annotations:
        video_id = ann.get("video_id", ann.get("image_id"))
        captions.setdefault(video_id, ann["caption"])
    return captions


def evaluate(video_embs: List[torch.Tensor], text_embs: List[torch.Tensor]) -> Dict[str, Dict[str, float]]:
    videos = torch.stack([emb.squeeze(0) if emb.ndim == 2 else emb for emb in video_embs])
    texts = torch.stack([emb.squeeze(0) if emb.ndim == 2 else emb for emb in text_embs])
    videos = videos / (videos.norm(dim=1, keepdim=True) + 1e-8)
    texts = texts / (texts.norm(dim=1, keepdim=True) + 1e-8)
    gt = list(range(len(video_embs)))
    sim_t2v = texts @ videos.T
    sim_v2t = videos @ texts.T
    return {
        "recall_t2v": {str(k): v for k, v in compute_recall(sim_t2v.numpy(), gt, TOP_K).items()},
        "recall_v2t": {str(k): v for k, v in compute_recall(sim_v2t.numpy(), gt, TOP_K).items()},
    }


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def encode_folder(encoder: PerceptionEncoder, folder: Path):
    return encoder.compute_video_emb(video_frame_folder=str(folder), return_frame_embs=True)


def run_uniform(
    *,
    name: str,
    counts: Dict[str, int],
    encoder: PerceptionEncoder,
    video_ids: List[str],
    captions: Dict[str, str],
    video_dir: Path,
    out_dir: Path,
) -> Dict:
    method_dir = out_dir / name
    reset_dir(method_dir)
    video_embs: List[torch.Tensor] = []
    text_embs: List[torch.Tensor] = []
    rows = []

    for video_id in video_ids:
        frame_count = max(1, int(counts[video_id]))
        video_path = video_dir / f"{video_id}.mp4"
        frame_dir = method_dir / "frames" / video_id
        sampler = FrameSampler_Uniform(no_of_frames=frame_count)

        _, sampling_sec = timed(lambda: sampler.extract_frames(str(video_path), str(frame_dir)))
        (frame_embs, video_emb), embedding_sec = timed(lambda: encode_folder(encoder, frame_dir))
        text_emb, text_sec = timed(lambda: encoder.compute_text_emb(captions[video_id]))

        video_embs.append(video_emb.squeeze(0))
        text_embs.append(text_emb.squeeze(0))
        rows.append(
            {
                "video_id": video_id,
                "requested_frames": frame_count,
                "encoded_frames": int(frame_embs.shape[0]),
                "selected_frames": int(frame_embs.shape[0]),
                "sampling_sec": sampling_sec,
                "embedding_sec": embedding_sec + text_sec,
                "scoring_sec": 0.0,
                "total_sec": sampling_sec + embedding_sec + text_sec,
            }
        )

    return summarize(name, rows, video_embs, text_embs)


def run_stacfp_keyscore(
    *,
    encoder: PerceptionEncoder,
    video_ids: List[str],
    captions: Dict[str, str],
    video_dir: Path,
    out_dir: Path,
    threshold: float,
) -> Dict:
    method_dir = out_dir / "stacfp_keyscore"
    reset_dir(method_dir)
    sampler = FrameSampler_STACFP(frame_step=5, gamma_time=5, k_min=6, k_max=30)
    scorer = FrameScorer(pooling=MeanPooling(), alpha=0.7, beta=0.1, gamma=0.2, normalize_each=True)

    video_embs: List[torch.Tensor] = []
    text_embs: List[torch.Tensor] = []
    rows = []

    for video_id in video_ids:
        video_path = video_dir / f"{video_id}.mp4"
        frame_dir = method_dir / "frames" / video_id

        saved_paths, sampling_sec = timed(lambda: sampler.extract_frames(str(video_path), str(frame_dir)))
        (frame_embs, _), embedding_sec = timed(lambda: encode_folder(encoder, frame_dir))
        text_emb, text_sec = timed(lambda: encoder.compute_text_emb(captions[video_id]))
        text_vec = text_emb.squeeze(0)

        scores, scoring_sec = timed(lambda: scorer.compute(frame_embs, text_vec)["scores"])
        mask = scores >= threshold
        if not bool(mask.any()):
            mask[int(torch.argmax(scores).item())] = True
        selected = frame_embs[mask]
        video_emb = mean_pool(selected).squeeze(0)

        video_embs.append(video_emb)
        text_embs.append(text_vec)
        rows.append(
            {
                "video_id": video_id,
                "candidate_frames": int(frame_embs.shape[0]),
                "saved_paths": len(saved_paths),
                "selected_frames": int(selected.shape[0]),
                "threshold": threshold,
                "sampling_sec": sampling_sec,
                "embedding_sec": embedding_sec + text_sec,
                "scoring_sec": scoring_sec,
                "total_sec": sampling_sec + embedding_sec + text_sec + scoring_sec,
            }
        )

    return summarize("stacfp_keyscore", rows, video_embs, text_embs)


def summarize(name: str, rows: List[Dict], video_embs: List[torch.Tensor], text_embs: List[torch.Tensor]) -> Dict:
    numeric_keys = ["sampling_sec", "embedding_sec", "scoring_sec", "total_sec"]
    avg = {key: float(np.mean([row[key] for row in rows])) for key in numeric_keys}
    total = {key: float(np.sum([row[key] for row in rows])) for key in numeric_keys}
    avg["encoded_frames"] = float(np.mean([row.get("encoded_frames", row.get("candidate_frames", 0)) for row in rows]))
    avg["selected_frames"] = float(np.mean([row["selected_frames"] for row in rows]))
    return {
        "name": name,
        "num_videos": len(rows),
        "average": avg,
        "total": total,
        "retrieval": evaluate(video_embs, text_embs),
        "per_video": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini MSR-VTT runtime benchmark for WACV revision.")
    parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT / "MSRVTT")
    parser.add_argument("--list", type=Path, default=DATA_ROOT / "MSRVTT/structured-symlinks/test_list_mini.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "wacv_runtime_mini")
    args = parser.parse_args()

    video_dir = args.dataset_root / "videos" / "all"
    captions = load_captions(args.dataset_root / "annotation" / "MSR_VTT.json")
    video_ids = load_ids(args.list, args.limit)
    missing = [video_id for video_id in video_ids if not (video_dir / f"{video_id}.mp4").exists()]
    if missing:
        raise FileNotFoundError(f"Missing videos: {missing}")

    args.out.mkdir(parents=True, exist_ok=True)
    encoder = PerceptionEncoder()

    results: Dict[str, Dict] = {
        "metadata": {
            "video_ids": video_ids,
            "threshold": args.threshold,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "note": "Mini-subset timing for pipeline verification; do not report as full 1k MSR-VTT result.",
        }
    }

    results["ufp8"] = run_uniform(
        name="ufp8",
        counts={video_id: 8 for video_id in video_ids},
        encoder=encoder,
        video_ids=video_ids,
        captions=captions,
        video_dir=video_dir,
        out_dir=args.out,
    )

    results["stacfp_keyscore"] = run_stacfp_keyscore(
        encoder=encoder,
        video_ids=video_ids,
        captions=captions,
        video_dir=video_dir,
        out_dir=args.out,
        threshold=args.threshold,
    )

    candidate_counts = {
        row["video_id"]: max(1, int(row["candidate_frames"]))
        for row in results["stacfp_keyscore"]["per_video"]
    }
    selected_counts = {
        row["video_id"]: max(1, int(row["selected_frames"]))
        for row in results["stacfp_keyscore"]["per_video"]
    }

    results["ufp_candidate_matched"] = run_uniform(
        name="ufp_candidate_matched",
        counts=candidate_counts,
        encoder=encoder,
        video_ids=video_ids,
        captions=captions,
        video_dir=video_dir,
        out_dir=args.out,
    )
    results["ufp_selected_matched"] = run_uniform(
        name="ufp_selected_matched",
        counts=selected_counts,
        encoder=encoder,
        video_ids=video_ids,
        captions=captions,
        video_dir=video_dir,
        out_dir=args.out,
    )

    out_json = args.out / "msrvtt_mini_runtime.json"
    with out_json.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps({key: val.get("average", val) for key, val in results.items()}, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()