"""
Offline frame embedding for MSVD — reads pre-extracted jpg frame directories.

MSVD stores videos as directories of numbered jpgs (001.jpg, 002.jpg, …) rather
than raw video files. This script walks each subdirectory under `frames_root`,
embeds all frames with ClipEncoder, and writes:
  {out_dir}/{video_stem}/frame_embeddings.pt   – [N, D] float32 tensor
  {out_dir}/{video_stem}/frame_paths.json       – list of absolute frame paths

The output format is identical to embed_frames.py so that FrameSelector and
msvd_qa_eval.py can consume it without changes.

Usage
-----
  python msvd_embed_frames.py \\
      --frames_root  $KEYSCORE_DATA_ROOT/MSVD/testing \\
      --out_dir      $KEYSCORE_DATA_ROOT/MSVD/embeddings_stacfp \\
      --video_ids    $KEYSCORE_DATA_ROOT/MSVD/test_stems.txt
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# ── resolve repo root ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from keyscore.encoders.clip_encoder import ClipEncoder  # noqa: E402


def embed_one_video_from_frames(
    frame_dir: Path,
    out_emb_dir: Path,
    encoder: ClipEncoder,
) -> bool:
    """
    Embed all .jpg frames in frame_dir and save frame_embeddings.pt and
    frame_paths.json under out_emb_dir. Returns True on success.
    """
    out_emb_dir.mkdir(parents=True, exist_ok=True)
    emb_pt = out_emb_dir / "frame_embeddings.pt"
    if emb_pt.exists():
        return True
    frame_paths = sorted(frame_dir.glob("*.jpg"), key=lambda p: p.name)
    if not frame_paths:
        return False
    frame_path_strs = [str(p) for p in frame_paths]
    try:
        frame_embs = encoder.embed_frames(frame_path_strs)
    except Exception:
        return False
    torch.save(frame_embs, emb_pt)
    (out_emb_dir / "frame_paths.json").write_text(
        json.dumps(frame_path_strs, indent=2)
    )
    return True


def embed_from_frame_dirs(
    frames_root: Path,
    out_dir: Path,
    video_stems: list[str] | None,
    batch_size: int,
) -> None:
    enc = ClipEncoder(batch_size=batch_size, amp=True, num_workers=4, prefetch_factor=2)
    out_dir.mkdir(parents=True, exist_ok=True)

    if video_stems is not None:
        dirs = [(s, frames_root / s) for s in video_stems]
    else:
        dirs = sorted(
            [(d.name, d) for d in frames_root.iterdir() if d.is_dir()],
            key=lambda x: x[0],
        )

    missing = [s for s, d in dirs if not d.exists()]
    if missing:
        print(f"[WARN] {len(missing)} frame directories not found, e.g.: {missing[:3]}")
    dirs = [(s, d) for s, d in dirs if d.exists()]

    print(f"Frame dirs : {len(dirs)}  →  {out_dir}")

    for stem, frame_dir in tqdm(dirs, desc="embed"):
        emb_pt = out_dir / stem / "frame_embeddings.pt"
        if emb_pt.exists():
            continue  # resume-safe

        frame_paths = sorted(frame_dir.glob("*.jpg"), key=lambda p: p.name)
        if not frame_paths:
            print(f"[WARN] no .jpg frames in {frame_dir}, skipping")
            continue

        frame_path_strs = [str(p) for p in frame_paths]
        try:
            frame_embs = enc.embed_frames(frame_path_strs)  # [N, D]
        except Exception as e:
            print(f"[WARN] embedding failed for {stem}: {e}")
            continue

        emb_dir = out_dir / stem
        emb_dir.mkdir(parents=True, exist_ok=True)
        torch.save(frame_embs, emb_pt)
        (emb_dir / "frame_paths.json").write_text(json.dumps(frame_path_strs, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed pre-extracted MSVD frame directories")
    ap.add_argument("--frames_root", required=True, type=Path,
                    help="Root dir containing per-video subdirs of .jpg frames")
    ap.add_argument("--out_dir",     required=True, type=Path,
                    help="Where to write {stem}/frame_embeddings.pt + frame_paths.json")
    ap.add_argument("--video_ids",   default=None, type=Path,
                    help="Optional text file listing video stems to process (one per line)")
    ap.add_argument("--batch_size",  default=512, type=int,
                    help="ClipEncoder image batch size (default: 512)")
    args = ap.parse_args()

    video_stems: list[str] | None = None
    if args.video_ids:
        video_stems = [l.strip() for l in args.video_ids.read_text().splitlines() if l.strip()]

    embed_from_frame_dirs(
        frames_root=args.frames_root,
        out_dir=args.out_dir,
        video_stems=video_stems,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
