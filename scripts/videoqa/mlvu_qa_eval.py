#!/usr/bin/env python3
"""
MLVU (Multi-task Long Video Understanding) evaluation for KeyScore.

We evaluate the *localizable-evidence* multiple-choice tasks of MLVU --
needle (findNeedle), anomaly_reco, count, order -- where the answer hinges on
a sparse set of decisive frames rather than holistic global reasoning. This is
the regime where decision-centric frame utility (drop-impact + temporal
coverage) is hypothesized to help most over a purely semantic top-K selector,
and complements the EgoSchema negative result (holistic egocentric reasoning,
where uniform sampling is already near-optimal).

Pipeline (mirrors the Video-MME / EgoSchema pipeline in this repo):
  1. Extract a uniform candidate pool of N frames per video (cv2).
  2. Embed candidates with ClipEncoder (CLIP-ViT-H).
  3. For each question, select K frames with each method:
       - uniform   : K evenly spaced frames (blind baseline, "UFP")
       - semantic  : FrameScorer with alpha=1 (Query-TopK, semantic-only)
       - keyscore  : FrameScorer S+T+D (alpha=0.7, beta=0.1, gamma=0.2)
  4. Answer the 4-way MCQ with Qwen2-VL.
  5. Score against the official MLVU annotations (answer is the text of the
     correct candidate; matched back to its index).

Usage
-----
  python mlvu_qa_eval.py \
      --mlvu_root $KEYSCORE_DATA_ROOT/MLVU \
      --out_dir   outputs/mlvu_keyscore \
      --tasks 2_needle 6_anomaly_reco 4_count 5_order \
      --per_task 40 --n_candidates 32 --k 8 \
      --model_id Qwen/Qwen2-VL-7B-Instruct
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

LETTERS = ["A", "B", "C", "D"]

# MLVU task json -> video subfolder (same name) mapping for MCQ tasks.
TASK_FOLDER = {
    "2_needle": "2_needle",
    "6_anomaly_reco": "6_anomaly_reco",
    "4_count": "4_count",
    "5_order": "5_order",
}


# ----------------------------------------------------------------------------
def _uniform_idx(total: int, k: int) -> list[int]:
    if k >= total:
        return list(range(total))
    if k == 1:
        return [total // 2]
    step = (total - 1) / (k - 1)
    return [round(i * step) for i in range(k)]


def extract_uniform_frames(video_path: Path, out_dir: Path, n: int) -> list[str]:
    """Extract n uniformly spaced frames from video_path as jpgs. Returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if len(existing) >= n:
        return [str(p) for p in existing[:n]]

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
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


def answer_mcq4(runner: QwenVLRunner, batch, max_new_tokens: int = 6) -> list[str]:
    """4-way MCQ (A-D). batch = [(frame_paths, question, options4), ...]."""
    preds = []
    bs = 4
    for i in range(0, len(batch), bs):
        chunk = batch[i:i + bs]
        messages_list = []
        for paths, question, options in chunk:
            opt_txt = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(options))
            prompt = (
                "You are watching a long video. The frames below are sampled "
                "from the video in temporal order.\n"
                f"Question: {question}\n"
                f"Options:\n{opt_txt}\n"
                "Reply with only the letter of the correct option (A, B, C, or D)."
            )
            messages_list.append([{
                "role": "user",
                "content": [{"type": "image", "image": str(p)} for p in paths]
                           + [{"type": "text", "text": prompt}],
            }])
        raw = runner._run_batch(messages_list, max_new_tokens)
        for r in raw:
            preds.append(_extract_letter4(r))
        torch.cuda.empty_cache()
    return preds


def _extract_letter4(text: str) -> str:
    import re
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else (text.strip()[:1].upper() or "A")


# ----------------------------------------------------------------------------
def build_subset(mlvu_root: Path, tasks: list[str], per_task: int) -> list[dict]:
    """Load the MCQ subset: per_task questions per task, with gt letter."""
    items = []
    for t in tasks:
        folder = TASK_FOLDER[t]
        data = json.loads((mlvu_root / "json" / f"{t}.json").read_text())
        n = 0
        for q in data:
            if "candidates" not in q or "answer" not in q:
                continue
            cands = q["candidates"]
            if q["answer"] not in cands:
                continue
            gt = LETTERS[cands.index(q["answer"])]
            items.append({
                "task": t,
                "folder": folder,
                "video": q["video"],
                "question": q["question"],
                "candidates": cands,
                "gt": gt,
                "uid": f"{t}/{q['video']}/{n}",
            })
            n += 1
            if n >= per_task:
                break
    return items


def run(mlvu_root: Path, out_dir: Path, tasks: list[str], per_task: int,
        n_candidates: int, k: int, model_id: str, cache_dir: Path,
        methods: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_root = mlvu_root / "videos"

    items = build_subset(mlvu_root, tasks, per_task)
    # keep only items whose video exists locally
    items = [it for it in items
             if (video_root / it["folder"] / it["video"]).exists()]
    print(f"[INFO] MLVU subset: {len(items)} questions across {tasks} "
          f"| n_candidates={n_candidates} | k={k}")

    scorer_keyscore = FrameScorer(MeanPooling(), alpha=0.7, beta=0.1, gamma=0.2)
    scorer_semantic = FrameScorer(MeanPooling(), alpha=1.0, beta=0.0, gamma=0.0)

    encoder = ClipEncoder(batch_size=256, amp=True, num_workers=4)
    runner = QwenVLRunner(model_id=model_id)

    ckpt = out_dir / "mlvu_checkpoint.jsonl"
    done = set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["uid"])
        print(f"[INFO] Resuming: {len(done)} already done")

    fout = open(ckpt, "a")
    for it in tqdm(items, desc="mlvu"):
        if it["uid"] in done:
            continue
        question = it["question"]
        options = it["candidates"]
        gt = it["gt"]

        # frame cache keyed by video file (shared across questions on same video)
        vid_key = f"{it['folder']}__{it['video']}"
        frame_paths = extract_uniform_frames(
            video_root / it["folder"] / it["video"],
            cache_dir / vid_key, n_candidates)
        if len(frame_paths) < k:
            tqdm.write(f"[WARN] {it['uid']}: only {len(frame_paths)} frames, skip")
            continue

        frame_embs = encoder.embed_frames(frame_paths)
        text_emb = encoder.embed_text(question)

        sel = {}
        if "uniform" in methods:
            sel["uniform"] = [frame_paths[i] for i in _uniform_idx(len(frame_paths), k)]
        if "semantic" in methods:
            sel["semantic"] = [frame_paths[i] for i in
                               select_topk(frame_embs, text_emb, scorer_semantic, k)]
        if "keyscore" in methods:
            sel["keyscore"] = [frame_paths[i] for i in
                               select_topk(frame_embs, text_emb, scorer_keyscore, k)]

        rec = {"uid": it["uid"], "task": it["task"], "gt": gt}
        batch = [(paths, question, options) for paths in sel.values()]
        preds = answer_mcq4(runner, batch)
        for m, pred in zip(sel.keys(), preds):
            rec[m] = {"pred": pred, "correct": int(pred == gt)}

        fout.write(json.dumps(rec) + "\n")
        fout.flush()
    fout.close()

    summarize(ckpt, out_dir, methods, tasks)


def summarize(ckpt: Path, out_dir: Path, methods: list[str],
              tasks: list[str]) -> None:
    overall = {m: [0, 0] for m in methods}
    per_task = {t: {m: [0, 0] for m in methods} for t in tasks}
    for line in ckpt.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        t = rec.get("task")
        for m in methods:
            if m in rec:
                overall[m][1] += 1
                overall[m][0] += rec[m]["correct"]
                if t in per_task:
                    per_task[t][m][1] += 1
                    per_task[t][m][0] += rec[m]["correct"]

    def fmt(counts):
        out = {}
        for m in methods:
            c, tt = counts[m]
            out[m] = {"correct": c, "total": tt,
                      "acc": (100.0 * c / tt if tt else 0.0)}
        return out

    summary = {"overall": fmt(overall),
               "per_task": {t: fmt(per_task[t]) for t in tasks}}

    print("\n=== MLVU accuracy (localizable-evidence MCQ tasks) ===")
    print("  [overall]")
    for m in methods:
        s = summary["overall"][m]
        print(f"    {m:10s}: {s['correct']}/{s['total']} = {s['acc']:.2f}%")
    for t in tasks:
        print(f"  [{t}]")
        for m in methods:
            s = summary["per_task"][t][m]
            print(f"    {m:10s}: {s['correct']}/{s['total']} = {s['acc']:.2f}%")
    (out_dir / "mlvu_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[INFO] Summary -> {out_dir / 'mlvu_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlvu_root", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--cache_dir", type=Path, default=None)
    ap.add_argument("--tasks", nargs="+",
                    default=["2_needle", "6_anomaly_reco", "4_count", "5_order"])
    ap.add_argument("--per_task", type=int, default=40)
    ap.add_argument("--n_candidates", type=int, default=32)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model_id", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--methods", nargs="+",
                    default=["uniform", "semantic", "keyscore"])
    ap.add_argument("--summarize_only", action="store_true")
    args = ap.parse_args()

    cache_dir = args.cache_dir or (args.mlvu_root / "frame_cache")
    if args.summarize_only:
        summarize(args.out_dir / "mlvu_checkpoint.jsonl", args.out_dir,
                  args.methods, args.tasks)
        return
    run(args.mlvu_root, args.out_dir, args.tasks, args.per_task,
        args.n_candidates, args.k, args.model_id, cache_dir, args.methods)


if __name__ == "__main__":
    main()
