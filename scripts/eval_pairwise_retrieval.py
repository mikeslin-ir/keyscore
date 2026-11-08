"""Protocol-correct, fixed-budget MSR-VTT retrieval with reproducible cached PE inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keyscore import KeyScore
from keyscore.scoring import METHODS, WEIGHTS

BACKBONE = "hf-hub:timm/PE-Core-bigG-14-448"
PROPOSALS = {
    "frame_step": 5, "gamma_time": 5.0, "k_min": 6, "k_max": 30,
    "color_space": "HSV", "hist_bins": [8, 8, 8],
    "resize_shorter_to": 256, "random_state": 42, "image_quality": 95,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def recall(scores: torch.Tensor) -> dict[str, float]:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or scores.shape[0] == 0:
        raise ValueError("The single-caption protocol requires a nonempty square score matrix.")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Retrieval scores must be finite.")
    order = scores.argsort(dim=1, descending=True, stable=True)
    targets = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
    return {
        str(k): float((order[:, :k] == targets).any(dim=1).float().mean() * 100)
        for k in (1, 5, 10)
    }


def metrics(scores: torch.Tensor) -> dict:
    return {"recall_t2v": recall(scores), "recall_v2t": recall(scores.T)}


def load_inputs(root: Path, list_path: Path, limit: int | None) -> tuple[list[str], list[str], Path]:
    ids = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("The video list contains duplicate IDs.")
    ids = ids if limit is None else ids[:limit]
    if not ids:
        raise ValueError("The video list is empty.")
    annotation_path = root / "annotation" / "MSR_VTT.json"
    annotations = json.loads(annotation_path.read_text())["annotations"]
    captions: dict[str, str] = {}
    for item in annotations:
        video_id = item.get("video_id", item.get("image_id"))
        captions.setdefault(video_id, item["caption"])
    for video_id in ids:
        if video_id not in captions:
            raise ValueError(f"Missing caption for {video_id}.")
        if not (root / "videos" / "all" / f"{video_id}.mp4").is_file():
            raise FileNotFoundError(f"Missing video: {video_id}")
    return ids, [captions[video_id] for video_id in ids], annotation_path


def prepare_cache(args, ids: list[str], captions: list[str], device: torch.device) -> list[dict]:
    from keyscore.proposals.stacfp import FrameSampler_STACFP

    args.cache.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            ROOT / "keyscore/encoders/perception_encoder.py",
            ROOT / "keyscore/proposals/stacfp.py",
        )
    }
    config = {"backbone": BACKBONE, "proposals": PROPOSALS, "source_sha256": source_hashes}
    encoder = None
    sampler = FrameSampler_STACFP(**PROPOSALS)
    records = []
    for i, (video_id, caption) in enumerate(zip(ids, captions)):
        video_path = args.dataset_root / "videos" / "all" / f"{video_id}.mp4"
        video_stat = video_path.stat()
        identity = {
            "video_id": video_id, "caption": caption, "config": config,
            "video_path": str(video_path.resolve()),
            "video_size": video_stat.st_size, "video_mtime_ns": video_stat.st_mtime_ns,
        }
        folder = args.cache / video_id
        path = folder / "embeddings.pt"
        if path.exists():
            record = torch.load(path, map_location="cpu", weights_only=True)
            if record["identity"] != identity:
                raise ValueError(f"Cache metadata mismatch: {path}; use a new cache directory.")
        else:
            if encoder is None:
                from keyscore.encoders.perception_encoder import PerceptionEncoder
                print(f"Loading cached {BACKBONE} on {device}.", flush=True)
                encoder = PerceptionEncoder(device=str(device))
            start = time.perf_counter()
            paths = sampler.extract_frames(str(video_path), str(folder / "frames"))
            sampling_sec = time.perf_counter() - start
            synchronize(device)
            start = time.perf_counter()
            frames = encoder.encode_images(paths, batch_size=args.encoder_batch_size)
            text = encoder.encode_text(caption).squeeze(0)
            synchronize(device)
            encoding_sec = time.perf_counter() - start
            if frames.ndim != 2 or frames.shape[0] != len(paths) or frames.shape[1] != text.numel():
                raise ValueError(f"Invalid encoder output for {video_id}.")
            if not bool(torch.isfinite(frames).all() and torch.isfinite(text).all()):
                raise ValueError(f"Nonfinite encoder output for {video_id}.")
            record = {
                "identity": identity, "frames": frames, "text": text,
                "frame_paths": paths, "frame_sha256": [sha256(Path(p)) for p in paths],
                "sampling_sec": sampling_sec, "encoding_sec": encoding_sec,
            }
            temporary = folder / "embeddings.pt.tmp"
            torch.save(record, temporary)
            temporary.replace(path)
        records.append(record)
        print(f"Prepared {i+1}/{len(ids)}: {video_id}, {record['frames'].shape[0]} candidates.", flush=True)
    return records


@torch.inference_mode()
def evaluate(records: list[dict], device: torch.device, k: int) -> tuple[dict, dict]:
    frames = [F.normalize(record["frames"].to(device).float(), dim=1) for record in records]
    queries = F.normalize(torch.stack([record["text"] for record in records]).to(device).float(), dim=1)
    mean_representations = torch.stack([F.normalize(f.mean(dim=0), dim=0) for f in frames])
    synchronize(device)
    start = time.perf_counter()
    mean_scores = (queries @ mean_representations.T).cpu()
    synchronize(device)
    mean_sec = time.perf_counter() - start
    matrices = {"candidate_mean": mean_scores}
    results = {"candidate_mean": {**metrics(mean_scores), "online_sec": mean_sec}}
    pair_details = {}
    oracle_scores = torch.empty_like(mean_scores)
    selector = KeyScore()
    for method in METHODS:
        columns, redundancies, selected = [], [], []
        synchronize(device)
        start = time.perf_counter()
        for j, f in enumerate(frames):
            selection = selector.select(f, queries, k, method=method)
            indices = selection.indices
            columns.append(selection.similarities)
            selected.append(indices)
            effective_k = indices.shape[1]
            if effective_k > 1:
                sums = f[indices].sum(dim=1)
                redundancies.append(
                    ((sums.square().sum(dim=1) - effective_k) / (effective_k * (effective_k - 1))).clamp(-1, 1)
                )
        synchronize(device)
        elapsed = time.perf_counter() - start
        matrix = torch.stack(columns, dim=1).cpu()
        matrices[method] = matrix
        results[method] = {
            **metrics(matrix), "online_sec": elapsed,
            "avg_selected_frames": sum(min(k, f.shape[0]) for f in frames) / len(frames),
            "avg_selected_pairwise_cosine": (
                float(torch.cat(redundancies).mean()) if redundancies else None
            ),
        }
        pair_details[method] = {
            "selected_indices_by_video": [index.cpu() for index in selected],
        }
        if method == "keyscore":
            # Isolate the legacy diagnostic from both proper scores and online timing.
            own_caption_means = torch.stack([
                F.normalize(f[index[j]].mean(dim=0), dim=0)
                for j, (f, index) in enumerate(zip(frames, selected))
            ])
            oracle_scores = (queries @ own_caption_means.T).cpu()
        print(f"{method}: {json.dumps(results[method])}", flush=True)
    diagnostic = {
        "own_caption_conditioned_keyscore": {
            **metrics(oracle_scores),
            "valid_standard_retrieval_result": False,
            "note": "Uses each video's matched test caption to select its shared representation.",
        }
    }
    full_hits = matrices["keyscore"].argmax(dim=1) == torch.arange(len(records))
    semantic_hits = matrices["semantic"].argmax(dim=1) == torch.arange(len(records))
    deltas = (full_hits.int() - semantic_hits.int()).numpy()
    rng = np.random.default_rng(42)
    bootstrap = rng.choice(deltas, size=(2000, len(deltas)), replace=True).mean(axis=1) * 100
    candidate_redundancy = [
        float(((f.sum(dim=0).square().sum() - f.shape[0]) / (f.shape[0] * (f.shape[0] - 1))).clamp(-1, 1))
        if f.shape[0] > 1 else None for f in frames
    ]
    return {
        "methods": results, "diagnostics": diagnostic,
        "paired_full_minus_semantic_t2v_r1": {
            "delta_percentage_points": float(deltas.mean() * 100),
            "bootstrap_95_percentile_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "seed": 42, "replicates": 2000, "unit": "video/single-caption query",
        },
        "candidate_pairwise_cosine_by_video": candidate_redundancy,
    }, {"score_matrices": matrices, "oracle_diagnostic_scores": oracle_scores, **pair_details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.k < 1 or args.encoder_batch_size < 1 or (args.limit is not None and args.limit < 2):
        parser.error("Require k>=1, encoder-batch-size>=1, and limit>=2 when provided.")
    if (args.out / "summary.json").exists():
        parser.error("This output already contains a completed run; choose a new output directory.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    torch.set_num_threads(4)
    ids, captions, annotation_path = load_inputs(args.dataset_root, args.list, args.limit)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = prepare_cache(args, ids, captions, device)
    preparation_sec = time.perf_counter() - started
    metadata = {
        "dataset": "MSR-VTT", "protocol": "all-query-video-pairs, first caption per video",
        "run_kind": "pilot" if args.limit is not None else "full-list-single-caption",
        "not_standard_all_caption_benchmark": True,
        "video_ids": ids, "num_videos": len(ids), "num_queries": len(ids),
        "num_query_video_pairs": len(ids) ** 2, "requested_k": args.k,
        "weights": WEIGHTS, "mmr_relevance": 0.7, "parameter_selection": "prespecified, no test tuning",
        "backbone": BACKBONE, "proposal_config": PROPOSALS,
        "list_path": str(args.list.resolve()), "list_sha256": sha256(args.list),
        "annotation_sha256": sha256(annotation_path), "captions": captions,
        "cache_path": str(args.cache.resolve()),
        "cache_sha256": {video_id: sha256(args.cache / video_id / "embeddings.pt") for video_id in ids},
        "preparation_wall_sec_this_run": preparation_sec,
        "cached_original_sampling_sec": sum(record["sampling_sec"] for record in records),
        "cached_original_encoding_sec": sum(record["encoding_sec"] for record in records),
        "hardware": {
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "architecture": platform.machine(), "torch": torch.__version__, "python": platform.python_version(),
            "gpu_process_snapshot": subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                text=True,
            ).splitlines() if device.type == "cuda" else [],
            "timing_status": "Exploratory; concurrent GPU activity is not excluded.",
        },
        "git_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                Path(__file__), ROOT / "keyscore/scoring/frame_scorer.py",
                ROOT / "keyscore/scoring/keyscore.py",
                ROOT / "keyscore/scoring/selection.py",
                ROOT / "keyscore/encoders/perception_encoder.py",
                ROOT / "keyscore/proposals/stacfp.py",
            )
        },
        "online_timing_scope": (
            "Cached embeddings already resident on device; selector, aggregation and pair scoring. "
            "Excludes retrieval ranking, result serialization, and the isolated oracle diagnostic."
        ),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    summary, details = evaluate(records, device, args.k)
    torch.save(details, args.out / "per_pair.pt")
    summary["metadata"] = metadata
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Completed: {args.out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
