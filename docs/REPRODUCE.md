# Runbook: revised KeyScore manuscript

Run commands from the repository root. This runbook covers the current
fixed-budget MSR-VTT and MLVU studies, not the historical threshold-based,
cross-backbone, Video-MME, action-classification, or summarization headlines.

## 1. Install and smoke-test

Use Python 3.10+ in a virtual environment, and install a suitable
[PyTorch build](https://pytorch.org/get-started/locally/).

```bash
python -m pip install -e ".[test]"
python -m pytest -q tests
python -m examples.score_frames
python -m examples.pairwise_retrieval
```

These examples use synthetic embeddings, require no models/datasets, and are
not benchmark accuracy demonstrations. Importing `KeyScore` loads no encoders.
For the following video experiments, additionally install:

```bash
python -m pip install -r requirements.txt
export KEYSCORE_DATA_ROOT=/path/to/datasets
```

The reference retrieval run used Python 3.10.14, PyTorch 2.12.0+cu130, and an
NVIDIA GB10. That is provenance, not a requirement or a runtime guarantee.
Models download on first use; access to their weights and dataset licenses
is the user's responsibility.

## 2. MSR-VTT inputs and exact operating point

Expected files:

```text
$KEYSCORE_DATA_ROOT/MSRVTT/
  videos/all/<video_id>.mp4
  annotation/MSR_VTT.json
  structured-symlinks/test_list_miech.txt
```

The full list must contain the 1,000 Miech test video IDs, without duplicates.
For each video the evaluator selects the first caption in annotation order.
This yields 1,000 queries and 1,000,000 query-video pairs, not an all-caption
benchmark. Input hashes are included in the released result summary.

Prespecified configuration:

- PE `hf-hub:timm/PE-Core-bigG-14-448`, frozen, 448-pixel input.
- STACFP: frame step 5, HSV 8x8x8, time weight 5, shorter side 256,
  cluster search 6--30, seed 42, ten K-means initializations.
- Final K=4, mean pooling of unit embeddings, then pooled normalization.
- Semantic `(1,0,0)`, S+D `(0.8,0,0.2)`, full `(0.7,0.1,0.2)`;
  MMR relevance 0.7. No test-time parameter sweep.

## 3. Pilot, then full retrieval

Start with 32 videos to check decoding, model access, memory, and output shape:

```bash
python scripts/eval_pairwise_retrieval.py \
  --dataset-root "$KEYSCORE_DATA_ROOT/MSRVTT" \
  --list "$KEYSCORE_DATA_ROOT/MSRVTT/structured-symlinks/test_list_miech.txt" \
  --cache outputs/cache/msrvtt-pe-stacfp \
  --out outputs/msrvtt-pilot32-k4 \
  --limit 32 --k 4 --device cuda
```

The pilot is not evidence for the full-list benchmark. For the full run, omit
`--limit` and choose a different output directory:

```bash
python scripts/eval_pairwise_retrieval.py \
  --dataset-root "$KEYSCORE_DATA_ROOT/MSRVTT" \
  --list "$KEYSCORE_DATA_ROOT/MSRVTT/structured-symlinks/test_list_miech.txt" \
  --cache outputs/cache/msrvtt-pe-stacfp \
  --out outputs/msrvtt-pairwise-k4 \
  --k 4 --device cuda
```

`--device cpu` is supported but encoding the full PE model is expensive.
`--encoder-batch-size` controls embedding memory.

Each cache entry records video identity, caption, backbone, proposal settings,
source hashes, decoded-frame hashes, and features. A metadata mismatch raises an
error: use a new cache directory rather than mixing old CLIP and PE features.
Validated entries can be reused after interruption. A completed output directory
is deliberately not overwritten.

Outputs:

- `metadata.json`: ordered inputs, protocol, configuration, hashes, and timing scope.
- `summary.json`: six methods, recalls, paired bootstrap, and an isolated diagnostic.
- `per_pair.pt`: six `[1000,1000]` score matrices and selected indices.

`own_caption_conditioned_keyscore` is an invalid standard-retrieval diagnostic,
not a benchmark result. Every proper score uses the current query to select
frames from every candidate video. V2T uses the transpose of that same matrix.

## 4. Analyze saved retrieval results

No decoding, model loading, or new selection is needed:

```bash
python scripts/analyze_pairwise_retrieval.py \
  --summary outputs/msrvtt-pairwise-k4/summary.json \
  --pairs outputs/msrvtt-pairwise-k4/per_pair.pt \
  --out outputs/msrvtt-pairwise-k4/manuscript_analysis.json
```

This computes candidate-redundancy thirds, full/semantic selected-set overlap,
and an exact paired McNemar test. Each stratum still ranks queries against the
**complete** 1k-video database. The analysis is descriptive and post-hoc, not
evidence of a causal redundancy effect or parameter tuning.

Compare with [the released summary](../results/msrvtt_pairwise_k4.json) and
[the result tables](RESULTS.md). CPU/GPU arithmetic, decoding, and dataset
versions can affect ties or results; an unexplained discrepancy should be
investigated rather than hidden behind a fixed tolerance.

## 5. MLVU: exact question slice, uniform candidates

Expected official layout: `MLVU/json/<task>.json` and
`MLVU/videos/<task>/<video>`. The four tasks are `2_needle`, `6_anomaly_reco`,
`4_count`, and `5_order`.

Install the Qwen runtime extras if needed:

```bash
python -m pip install accelerate qwen-vl-utils
```

Use the released **1,018-question manifest**, rather than the script's default
40-question-per-task pilot:

```bash
python scripts/videoqa/mlvu_qa_eval.py \
  --mlvu_root "$KEYSCORE_DATA_ROOT/MLVU" \
  --out_dir outputs/mlvu-manuscript \
  --cache_dir outputs/cache/mlvu-uniform32 \
  --question_ids results/mlvu_question_ids.json \
  --tasks 2_needle 6_anomaly_reco 4_count 5_order \
  --n_candidates 32 --k 8 \
  --methods uniform semantic keyscore \
  --model_id Qwen/Qwen2-VL-7B-Instruct
```

Manifest IDs refer to task/video/eligible-question-index in annotation order.
Missing IDs or videos fail explicitly. All arms share 32 **uniform** candidates,
CLIP-ViT-H/14 selector embeddings, the question text, the QA prompt, and eight
frames in temporal order. Answer options are given to Qwen, not to the selector.
This study isolates selection, not STACFP proposal quality.

New runs store `run_config.json` and append to `mlvu_checkpoint.jsonl`; resuming
requires the same configuration. A legacy checkpoint without a run config can
be summarized, but must not be silently resumed as a new configured experiment.
Keep frame caches specific to their source videos and candidate settings.

To recompute totals and exact paired McNemar tests from saved predictions:

```bash
python scripts/videoqa/mlvu_qa_eval.py \
  --mlvu_root "$KEYSCORE_DATA_ROOT/MLVU" \
  --out_dir outputs/mlvu-manuscript \
  --summarize_only
```

The released MLVU numbers are recomputed from prior paired predictions, not
fresh inference in this code-release task. New inference uses deterministic
candidate-index ties through the public API; model/runtime differences can
still change generated answers. Question-level tests do not correct for
dependence between questions about the same video.

## 6. Timing, limits, and release checks

- Four selected frames versus 7.703 candidates does not save the encodings
  already computed for those candidates.
- Online retrieval timings exclude decoding/encoding, ranking, and serialization;
  they are exploratory single passes, not an isolated latency benchmark.
- There is no completed current-protocol Video-MME comparison or frame-budget
  sweep in this release. Do not use historical tables as replacements.
- `KeyScore` rejects invalid weights and embeddings. `FrameScorer` remains for
  backwards compatibility, with its historical defaults and output names.
- Keep videos, weights, feature caches, credentials, and raw run directories
  out of a source release. Inspect your changes before committing or pushing.

For expected numbers and caveats, see [RESULTS.md](RESULTS.md).
