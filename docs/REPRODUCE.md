# Reproduction checklist

End-to-end steps to reproduce every table in the paper. See the
[README](../README.md#environment-setup) for environment setup first.

## 0. Preconditions

```bash
conda activate keyscore
export KEYSCORE_DATA_ROOT=/path/to/your/datasets
cd /path/to/keyscore
```

Sanity check:

```bash
python -c "
import torch
from keyscore.scoring import FrameScorer, MeanPooling
from keyscore.proposals import FrameSampler_STACFP
print('cuda:', torch.cuda.is_available())
print('scorer ok:', FrameScorer(MeanPooling(), 0.7, 0.1, 0.2).compute(
    torch.randn(12, 256), torch.randn(256))['scores'].shape)
"
```

## 1. Datasets

Only the datasets you intend to evaluate are required. None are redistributed here;
obtain them from their official sources under their own licenses.

| Dataset | Used for | Expected files under `$KEYSCORE_DATA_ROOT` |
| --- | --- | --- |
| MSR-VTT | Retrieval, runtime, ablations | `MSRVTT/videos/all/`, `MSRVTT/annotation/MSR_VTT.json`, `MSRVTT/structured-symlinks/test_list_miech.txt` |
| MSVD | Retrieval, MSVD-QA | `MSVD/YouTubeClips/`, `MSVD/AllVideoDescriptions.txt`, `MSVD/annotaions/msvd_data/test_list.txt` |
| DiDeMo | Retrieval | `DiDeMo/video/test/`, `DiDeMo/didemo_test.json` |
| MLVU | Long-video QA | `MLVU/` (official layout) |
| EgoSchema | Long-video QA | `EgoSchema/` (official layout, `subset_answers.json`) |

Backbone weights download automatically from the Hugging Face Hub on first use
(`timm/PE-Core-bigG-14-448`, `Qwen/Qwen2-VL-7B-Instruct`). There are no KeyScore
checkpoints — the method is training-free.

## 2. Hardware and expected cost

Reference runs used a single A100 80GB.

| Experiment | Scale | Rough wall-clock |
| --- | --- | --- |
| Retrieval, one dataset | 1k videos | ~1 h (base) + minutes (scoring) |
| Runtime benchmark | 1k videos | ~1 h |
| Query-aware ablation | 1k videos | ~1 h |
| `gamma_time` sweep | 300 videos x 6 settings | ~2 h |
| MLVU QA | 1018 questions | several hours (Qwen2-VL 7B dominates) |

QA scripts write `*_checkpoint.jsonl` and resume automatically, so long runs can be
interrupted safely.

## 3. Table-by-table mapping

| Paper table | Command | Reference output |
| --- | --- | --- |
| Component ablation | `scripts/retrieval/{msrvtt,msvd,didemo}_eval_base.py` then `*_eval_scoring.py` | – |
| Zero-shot retrieval | same as above | – |
| Video-MME | `scripts/videoqa/` + Qwen2-VL | – |
| Runtime / compute fairness | `scripts/benchmark_msrvtt_mini_runtime.py` | `results/msrvtt_runtime_1k.json` |
| Query-aware comparison | `scripts/eval_query_aware_baseline.py` | `results/msrvtt_query_aware.json` |
| `gamma_time` sensitivity | `scripts/eval_gamma_time_sweep.py` | `results/msrvtt_gamma_time_sweep.json` |
| MLVU per-task study | `scripts/videoqa/mlvu_qa_eval.py` | `results/mlvu_summary.json` |

## 4. Step-by-step

### 4.1 Retrieval (MSR-VTT / MSVD / DiDeMo)

Two stages so the expensive embedding pass is cached:

```bash
# Stage 1 — STACFP/UFP/SCFP sampling + PE embedding
python scripts/retrieval/msrvtt_eval_base.py

# Stage 2 — KeyScore scoring, selection, and R@1/5/10
python scripts/retrieval/msrvtt_eval_scoring.py
```

Stage 1 writes to `_msrvtt/sampled_frames/*` and `_msrvtt/embeddings/*`. Rerunning
stage 2 with different `alpha/beta/gamma` reuses those embeddings, which is how the
component ablation was produced.

Repeat with `msvd_*` and `didemo_*`.

### 4.2 Runtime and compute fairness

```bash
python scripts/benchmark_msrvtt_mini_runtime.py \
    --list $KEYSCORE_DATA_ROOT/MSRVTT/structured-symlinks/test_list_miech.txt \
    --out outputs/runtime_1k
```

Compare `outputs/runtime_1k/msrvtt_mini_runtime.json` against
`results/msrvtt_runtime_1k.json`. Absolute seconds depend on your GPU, storage and
decode throughput; the *relative* ordering and the frame counts should match.

### 4.3 Controlled query-aware ablation

```bash
python scripts/eval_query_aware_baseline.py --threshold 0.5 --out outputs/query_aware
```

Verifies that the gain comes from selection rather than from merely conditioning on
the query: all arms share the backbone and the candidate pool, and `QueryTopK` is
frame-matched to KeyScore.

### 4.4 STACFP sensitivity

```bash
python scripts/eval_gamma_time_sweep.py \
    --gammas 0 1 2 5 10 20 --limit 300 --out outputs/gamma_sweep
```

### 4.5 Long-video QA

```bash
python scripts/videoqa/mlvu_qa_eval.py \
    --mlvu_root $KEYSCORE_DATA_ROOT/MLVU \
    --out_dir outputs/mlvu_keyscore \
    --tasks 2_needle 6_anomaly_reco 4_count 5_order \
    --n_candidates 32 --k 8 \
    --model_id Qwen/Qwen2-VL-7B-Instruct

python scripts/videoqa/egoschema_qa_eval.py \
    --ego_root $KEYSCORE_DATA_ROOT/EgoSchema \
    --out_dir outputs/egoschema_keyscore \
    --n_candidates 32 --k 8 \
    --model_id Qwen/Qwen2-VL-7B-Instruct
```

MSVD-QA needs cached embeddings first:

```bash
python scripts/videoqa/msvd_embed_frames.py \
    --frames_root $KEYSCORE_DATA_ROOT/MSVD/testing \
    --out_dir     $KEYSCORE_DATA_ROOT/MSVD/embeddings_stacfp \
    --video_ids   $KEYSCORE_DATA_ROOT/MSVD/test_stems.txt

python scripts/videoqa/msvd_qa_eval.py --out_dir outputs/msvd_qa
```

## 5. Expected tolerances

- **Retrieval R@1** should match exactly given identical videos, decode library and
  test list. Differences of >0.5 point usually indicate a different test split or a
  different OpenCV/FFmpeg build changing which frames are decoded.
- **ASF (average selected frames)** is deterministic given the same threshold and
  STACFP settings (KMeans uses `random_state=42`).
- **QA accuracy** varies by a few tenths of a point across runs because Qwen2-VL
  generation is not bit-exact across hardware and precision settings.
- **Wall-clock timings** are hardware-dependent; only the relative comparison between
  the four rows of the runtime table is meaningful.

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `alpha + beta + gamma must sum to 1.0` | Rescale the weights passed to `FrameScorer`. |
| `Cannot open video: ...` | OpenCV lacks a codec; install `opencv-python` with FFmpeg support or re-encode the clip. |
| BFloat16 overflow when encoding text | Use `ClipEncoder(amp=False)`. |
| CUDA OOM during embedding | Lower the encoder `batch_size`, or raise STACFP `frame_step`. |
| Very slow sampling | Raise `frame_step`, or lower `resize_shorter_to` in `FrameSampler_STACFP`. |
| Few or single candidate frames | Video is nearly static; lower `k_min` or reduce `gamma_time`. |
