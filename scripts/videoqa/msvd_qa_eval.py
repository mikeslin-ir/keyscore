#!/usr/bin/env python3
"""
MSVD-QA KeyScore evaluation (open-ended). Uses pre-computed embeddings.
Configs: scored K=2,4 and uniform K=8. Writes per-video checkpoint for --resume.
Embedding folders: emb_dir / vid{video_id} (e.g. vid1451).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from frame_selector import FrameSelector, get_uniform_frame_paths
from qwen_vl_runner import QwenVLRunner


def load_qa(qa_file: Path) -> dict[int, list[dict]]:
    with open(qa_file) as f:
        data = json.load(f)
    by_video = defaultdict(list)
    for item in data:
        by_video[item["video_id"]].append({
            "question": item["question"],
            "answer": item["answer"],
            "id": item.get("id"),
        })
    return dict(by_video)


def _ckpt_path(out_dir: Path) -> Path:
    return out_dir / "msvd_qa_checkpoint.jsonl"


def load_checkpoint(out_dir: Path) -> tuple[set[int], dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _ckpt_path(out_dir)
    if not ckpt.exists():
        return set(), {}
    done = set()
    results = {}
    with open(ckpt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vid = rec.get("video_id")
            if vid is not None:
                done.add(vid)
            mode = rec.get("mode")
            k = rec.get("k")
            if vid is not None and mode is not None and k is not None:
                results.setdefault(vid, {})[f"{mode}_k{k}"] = rec
    return done, results


def append_checkpoint(out_dir: Path, video_id: int, per_mode_k: dict) -> None:
    ckpt = _ckpt_path(out_dir)
    with open(ckpt, "a") as f:
        for key, rec in per_mode_k.items():
            rec["video_id"] = video_id
            f.write(json.dumps(rec) + "\n")


def run(
    msvd_root: Path,
    qa_file: Path,
    emb_dir: Path,
    out_dir: Path,
    scored_k_list: list[int],
    uniform_k_list: list[int],
    modes: list[str],
    model_id: str,
    resume: bool,
    max_questions_per_video: int | None = None,
    frames_root: Path | None = None,
    embed_missing: bool = False,
    video_fraction: float = 1.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        ckpt = _ckpt_path(out_dir)
        if ckpt.exists():
            ckpt.unlink()
            print(f"[INFO] Cleared checkpoint for fresh run: {ckpt}")
    by_video = load_qa(qa_file)
    all_videos = sorted(by_video.keys())
    done, _ = load_checkpoint(out_dir) if resume else (set(), {})
    remaining = [v for v in all_videos if v not in done]
    if video_fraction < 1.0:
        n_take = max(1, int(len(remaining) * video_fraction))
        remaining = remaining[:n_take]
        print(f"[INFO] Using first {len(remaining)} videos (video_fraction={video_fraction})")
    print(f"[INFO] MSVD-QA test: {sum(len(q) for q in by_video.values())} questions, {len(all_videos)} videos")
    print(f"[INFO] Total: {len(all_videos)} | Done: {len(done)} | Remaining: {len(remaining)}")
    print(f"    scored K values: {scored_k_list}")
    print(f"   uniform K values: {uniform_k_list}")

    selector = FrameSelector()
    runner = QwenVLRunner(model_id=model_id)
    _embed_encoder = None
    _embed_one_from_frames = None

    for video_id in tqdm(remaining, desc="videos"):
        vid_emb_dir = emb_dir / f"vid{video_id}"
        if not (vid_emb_dir / "frame_embeddings.pt").exists():
            if embed_missing and frames_root is not None:
                frame_dir = frames_root / f"vid{video_id}"
                if frame_dir.is_dir():
                    if _embed_encoder is None:
                        from keyscore.encoders.clip_encoder import ClipEncoder
                        from msvd_embed_frames import embed_one_video_from_frames
                        _embed_encoder = ClipEncoder(batch_size=512, amp=True, num_workers=0)
                        _embed_one_from_frames = embed_one_video_from_frames
                    ok = _embed_one_from_frames(frame_dir, vid_emb_dir, _embed_encoder)
                    if not ok:
                        tqdm.write(f"[WARN] Skip vid{video_id}: embedding generation failed")
                        continue
                else:
                    tqdm.write(f"[WARN] Skip vid{video_id}: no embeddings and frame dir not found {frame_dir}")
                    continue
            else:
                tqdm.write(f"[WARN] Skip vid{video_id}: no embeddings (use --frames_root and --embed_missing to generate)")
                continue

        # Load once per video (avoids repeated I/O and speeds up eval)
        frame_embs = torch.load(vid_emb_dir / "frame_embeddings.pt", weights_only=True)
        with open(vid_emb_dir / "frame_paths.json") as f:
            frame_paths = json.load(f)

        questions = by_video[video_id]
        if max_questions_per_video is not None:
            questions = questions[:max_questions_per_video]
        per_mode_k = {}
        text_embs = (
            selector.encoder.embed_text([qa["question"] for qa in questions])
            if "scored" in modes else None
        )

        for mode in modes:
            k_list = scored_k_list if mode == "scored" else uniform_k_list
            for k in k_list:
                if mode == "uniform":
                    paths = get_uniform_frame_paths(frame_paths, k)
                    batch_frames = [paths] * len(questions)
                else:
                    batch_frames = [
                        selector.select_from_arrays_with_text_emb(
                            frame_embs, frame_paths, text_embs[i], k
                        )[0]
                        for i in range(len(questions))
                    ]

                batch_input = [(batch_frames[i], questions[i]["question"]) for i in range(len(questions))]
                preds = runner.answer_open_batch(batch_input)

                per_mode_k[f"{mode}_k{k}"] = {
                    "mode": mode,
                    "k": k,
                    "records": [
                        {
                            "id": questions[i].get("id"),
                            "question": questions[i]["question"],
                            "answer": questions[i]["answer"],
                            "prediction": preds[i],
                            "frame_paths": batch_frames[i],
                        }
                        for i in range(len(questions))
                    ],
                }

        append_checkpoint(out_dir, video_id, per_mode_k)

    _, all_results = load_checkpoint(out_dir)
    summary = {}
    for vid, recs in all_results.items():
        for key, rec in recs.items():
            mode, k = rec["mode"], rec["k"]
            summary.setdefault(f"{mode}_top{k}", {"correct": 0, "total": 0})
            for r in rec["records"]:
                summary[f"{mode}_top{k}"]["total"] += 1
                # Open-ended: exact match for simplicity (or use GPT scorer later)
                if (r["prediction"] or "").strip().lower() == (r["answer"] or "").strip().lower():
                    summary[f"{mode}_top{k}"]["correct"] += 1

    for key in sorted(summary.keys()):
        s = summary[key]
        acc = 100.0 * s["correct"] / s["total"] if s["total"] else 0.0
        print(f"  {key}: {s['correct']}/{s['total']} = {acc:.2f}%")
    summary_path = out_dir / "msvd_qa_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[INFO] Summary written to {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msvd_root", required=True, type=Path)
    ap.add_argument("--qa_file", required=True, type=Path)
    ap.add_argument("--emb_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--scored_k", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--uniform_k", nargs="+", type=int, default=[8])
    ap.add_argument("--modes", nargs="+", default=["scored", "uniform"])
    ap.add_argument("--model_id", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_questions_per_video", type=int, default=None)
    ap.add_argument("--frames_root", type=Path, default=None, help="Root of per-video frame dirs (e.g. .../testing) for --embed_missing")
    ap.add_argument("--embed_missing", action="store_true", help="Generate embeddings for videos that don't have them (requires --frames_root)")
    ap.add_argument("--video_fraction", type=float, default=1.0, help="Use only this fraction of remaining videos (e.g. 0.25 = first 1/4)")
    args = ap.parse_args()
    run(
        msvd_root=args.msvd_root,
        qa_file=args.qa_file,
        emb_dir=args.emb_dir,
        out_dir=args.out_dir,
        scored_k_list=args.scored_k,
        uniform_k_list=args.uniform_k,
        modes=args.modes,
        model_id=args.model_id,
        resume=args.resume,
        max_questions_per_video=args.max_questions_per_video,
        frames_root=args.frames_root,
        embed_missing=args.embed_missing,
        video_fraction=args.video_fraction,
    )


if __name__ == "__main__":
    main()
