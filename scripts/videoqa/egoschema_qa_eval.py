#!/usr/bin/env python3
"""
EgoSchema (long-form egocentric video QA) evaluation for KeyScore.

EgoSchema is a 5-way MCQ benchmark over ~3-minute egocentric clips, where the
correct answer typically requires aggregating evidence across a long temporal
span (high redundancy). This is the regime where decision-centric frame utility
(drop-impact + temporal coverage) is hypothesized to help most over a purely
semantic top-K selector.

Pipeline (mirrors the Video-MME pipeline in this repo):
  1. Extract a uniform candidate pool of N frames per video (cv2).
  2. Embed candidates with ClipEncoder (CLIP-ViT-H, same as Video-MME selection).
  3. For each question, select K frames with each method:
       - uniform   : K evenly spaced frames (blind baseline, "UFP")
       - semantic  : FrameScorer with alpha=1 (Query-TopK, semantic-only)
       - keyscore  : FrameScorer S+T+D (alpha=0.7, beta=0.1, gamma=0.2)
  4. Answer the 5-way MCQ with Qwen2-VL.
  5. Score against subset_answers.json (official 500-question labeled subset).

Usage
-----
  python egoschema_qa_eval.py \
      --ego_root $KEYSCORE_DATA_ROOT/EgoSchema \
      --out_dir  outputs/egoschema_keyscore \
      --n_candidates 32 --k 8 \
      --model_id Qwen/Qwen2-VL-7B-Instruct \
      --max_videos 50            # quick directional run; omit for full 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from keyscore.scoring.frame_scorer import FrameScorer, MeanPooling
from keyscore.encoders.clip_encoder import ClipEncoder
from qwen_vl_runner import QwenVLRunner

LETTERS = ["A", "B", "C", "D", "E"]


# ----------------------------------------------------------------------------
def extract_uniform_frames(video_path: Path, out_dir: Path, n: int) -> list[str]:
    """Extract n uniformly spaced frames from video_path as jpgs. Returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if len(existing) >= n:
        return [str(p) for p in existing[:n]]

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Fallback: sequential read
        frames = []
        ok, img = cap.read()
        while ok:
            frames.append(img)
            ok, img = cap.read()
        cap.release()
        total = len(frames)
        if total == 0:
            return []
        idxs = _uniform_idx(total, n)
        paths = []
        for j, i in enumerate(idxs):
            p = out_dir / f"frame_{j:03d}.jpg"
            cv2.imwrite(str(p), frames[i])
            paths.append(str(p))
        return paths

    idxs = _uniform_idx(total, n)
    paths = []
    for j, i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok or img is None:
            continue
        p = out_dir / f"frame_{j:03d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    cap.release()
    return paths


def _uniform_idx(total: int, k: int) -> list[int]:
    if k >= total:
        return list(range(total))
    if k == 1:
        return [total // 2]
    step = (total - 1) / (k - 1)
    return [round(i * step) for i in range(k)]


def select_topk(frame_embs: torch.Tensor, text_emb: torch.Tensor,
                scorer: FrameScorer, k: int) -> list[int]:
    """Return sorted (temporal-order) indices of top-k frames by scorer."""
    if text_emb.dim() == 2:
        text_emb = text_emb.squeeze(0)
    out = scorer.compute(frame_embs, text_emb)
    scores = out["scores"]
    k_eff = min(k, scores.shape[0])
    idx = scores.topk(k_eff).indices.tolist()
    return sorted(idx)


def answer_mcq5(runner: QwenVLRunner, batch, max_new_tokens: int = 6) -> list[str]:
    """5-way MCQ (A-E). batch = [(frame_paths, question, options5), ...]."""
    preds = []
    bs = 4
    for i in range(0, len(batch), bs):
        chunk = batch[i:i + bs]
        messages_list = []
        for paths, question, options in chunk:
            opt_txt = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(options))
            prompt = (
                "You are watching an egocentric video. The frames below are sampled "
                "from the video in temporal order.\n"
                f"Question: {question}\n"
                f"Options:\n{opt_txt}\n"
                "Reply with only the letter of the correct option (A, B, C, D, or E)."
            )
            messages_list.append([{
                "role": "user",
                "content": [{"type": "image", "image": str(p)} for p in paths]
                           + [{"type": "text", "text": prompt}],
            }])
        raw = runner._run_batch(messages_list, max_new_tokens)
        for r in raw:
            preds.append(_extract_letter5(r))
        torch.cuda.empty_cache()
    return preds


def _extract_letter5(text: str) -> str:
    import re
    m = re.search(r"\b([A-E])\b", text.upper())
    return m.group(1) if m else (text.strip()[:1].upper() or "A")


# ----------------------------------------------------------------------------
def run(ego_root: Path, out_dir: Path, n_candidates: int, k: int,
        model_id: str, max_videos: int | None, cache_dir: Path,
        methods: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    answers = json.loads((ego_root / "subset_answers.json").read_text())
    questions = {q["q_uid"]: q for q in json.loads((ego_root / "questions.json").read_text())}
    video_dir = ego_root / "videos" / "good_clips_git"

    uids = [u for u in answers if (video_dir / f"{u}.mp4").exists() and u in questions]
    uids.sort()
    if max_videos is not None:
        uids = uids[:max_videos]
    print(f"[INFO] EgoSchema subset: {len(uids)} questions | n_candidates={n_candidates} | k={k}")

    # Scorers
    scorer_keyscore = FrameScorer(MeanPooling(), alpha=0.7, beta=0.1, gamma=0.2)
    scorer_semantic = FrameScorer(MeanPooling(), alpha=1.0, beta=0.0, gamma=0.0)

    encoder = ClipEncoder(batch_size=256, amp=True, num_workers=4)
    runner = QwenVLRunner(model_id=model_id)

    ckpt = out_dir / "egoschema_checkpoint.jsonl"
    done = set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["q_uid"])
        print(f"[INFO] Resuming: {len(done)} already done")

    fout = open(ckpt, "a")
    for uid in tqdm(uids, desc="egoschema"):
        if uid in done:
            continue
        q = questions[uid]
        question = q["question"]
        options = [q[f"option {j}"] for j in range(5)]
        gt = LETTERS[answers[uid]]

        # 1. candidate frames
        frame_paths = extract_uniform_frames(
            video_dir / f"{uid}.mp4", cache_dir / uid, n_candidates)
        if len(frame_paths) < k:
            tqdm.write(f"[WARN] {uid}: only {len(frame_paths)} frames, skip")
            continue

        # 2. embed candidates
        frame_embs = encoder.embed_frames(frame_paths)  # [N, D]
        text_emb = encoder.embed_text(question)

        # 3. selections
        sel = {}
        if "uniform" in methods:
            sel["uniform"] = [frame_paths[i] for i in _uniform_idx(len(frame_paths), k)]
        if "semantic" in methods:
            sel["semantic"] = [frame_paths[i] for i in
                               select_topk(frame_embs, text_emb, scorer_semantic, k)]
        if "keyscore" in methods:
            sel["keyscore"] = [frame_paths[i] for i in
                               select_topk(frame_embs, text_emb, scorer_keyscore, k)]

        # 4. answer
        rec = {"q_uid": uid, "gt": gt}
        batch = [(paths, question, options) for paths in sel.values()]
        preds = answer_mcq5(runner, batch)
        for m, pred in zip(sel.keys(), preds):
            rec[m] = {"pred": pred, "correct": int(pred == gt)}

        fout.write(json.dumps(rec) + "\n")
        fout.flush()
    fout.close()

    # 5. summarize
    summarize(ckpt, out_dir, methods)


def summarize(ckpt: Path, out_dir: Path, methods: list[str]) -> None:
    counts = {m: [0, 0] for m in methods}
    for line in ckpt.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for m in methods:
            if m in rec:
                counts[m][1] += 1
                counts[m][0] += rec[m]["correct"]
    print("\n=== EgoSchema accuracy ===")
    summary = {}
    for m in methods:
        c, t = counts[m]
        acc = 100.0 * c / t if t else 0.0
        summary[m] = {"correct": c, "total": t, "acc": acc}
        print(f"  {m:10s}: {c}/{t} = {acc:.2f}%")
    (out_dir / "egoschema_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[INFO] Summary -> {out_dir / 'egoschema_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego_root", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--cache_dir", type=Path, default=None,
                    help="Where to cache extracted candidate frames (default: ego_root/frame_cache)")
    ap.add_argument("--n_candidates", type=int, default=32)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model_id", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--max_videos", type=int, default=None)
    ap.add_argument("--methods", nargs="+",
                    default=["uniform", "semantic", "keyscore"])
    ap.add_argument("--summarize_only", action="store_true")
    args = ap.parse_args()

    cache_dir = args.cache_dir or (args.ego_root / "frame_cache")
    if args.summarize_only:
        summarize(args.out_dir / "egoschema_checkpoint.jsonl", args.out_dir, args.methods)
        return
    run(args.ego_root, args.out_dir, args.n_candidates, args.k,
        args.model_id, args.max_videos, cache_dir, args.methods)


if __name__ == "__main__":
    main()
