# Detailed results

All results are **zero-shot** with frozen backbones. KeyScore adds no trainable
parameters.

Unless stated otherwise: `alpha=0.7, beta=0.1, gamma=0.2`, threshold `tau=0.5`,
STACFP `frame_step=5, gamma_time=5, k_min=6, k_max=30`.

---

## 1. Component ablation

Text-to-video (T2V) / video-to-text (V2T) R@1 (%) and average selected frames (ASF),
on top of the frozen PE-Core-G Video backbone.
S = semantic relevance, T = candidate-context representativeness, D = frame-drop contribution.

| Method | MSR-VTT T2V | V2T | ASF | MSVD T2V | V2T | ASF | DiDeMo T2V | V2T | ASF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PE-Core-G Video | 49.7 | 48.2 | 6.00 | 60.4 | 82.3 | 5.60 | 45.1 | 46.1 | 11.30 |
| + KeyScore (S) | 63.2 | 60.0 | 2.00 | 88.5 | 86.5 | 5.00 | 57.8 | 59.0 | 3.00 |
| + KeyScore (T) | 49.8 | 48.9 | 8.20 | 84.6 | 86.1 | 4.00 | 48.5 | 50.1 | 2.00 |
| + KeyScore (D) | 62.6 | 59.4 | 3.00 | 85.8 | 86.5 | 6.00 | 57.2 | 58.0 | 2.00 |
| + KeyScore (S+T) | 61.3 | 59.5 | 3.30 | 87.9 | 88.6 | 2.00 | 59.4 | 60.3 | 2.00 |
| + KeyScore (D+T) | 61.4 | 59.1 | 2.57 | 87.9 | 89.2 | 4.00 | 59.7 | 60.1 | 2.00 |
| + KeyScore (S+D) | 63.5 | 60.3 | 2.69 | 89.1 | 89.7 | 2.00 | 59.8 | 60.3 | 2.00 |
| **+ KeyScore (S+T+D)** | **63.9** | **60.5** | **2.50** | **89.2** | **89.2** | **2.00** | **60.4** | **60.3** | **2.00** |

Takeaways:

- **S** and **D** are the strong individual cues; **T** alone contributes little but
  reinforces either of the other two.
- On multi-moment DiDeMo, adding T and D lifts T2V R@1 from 57.8 (semantic-only,
  i.e. `alpha=1`, which reduces to relevance-only top-K) to 60.4 while using
  **fewer** frames (3 -> 2). The gain is therefore not a frame-budget artifact.
- The full combination is best on every dataset while using 2–2.5 frames.

## 2. Zero-shot video–text retrieval

| Model | MSR-VTT T2V | V2T | MSVD T2V | V2T | DiDeMo T2V | V2T |
| --- | --- | --- | --- | --- | --- | --- |
| CLIP4Clip | 32.0 | – | 45.2 | 48.4 | – | – |
| X-CLIP | 49.3 | 48.9 | 50.4 | 66.8 | 47.8 | 47.8 |
| UMT-L | 40.7 | 37.1 | 49.0 | 74.5 | 49.9 | 59.7 |
| SigLIP2-L/16 | 41.5 | 31.4 | 53.7 | 74.2 | 18.4 | – |
| InternVL | 44.7 | 40.2 | 43.4 | 67.6 | – | – |
| InternVideo2 | 51.9 | 50.9 | – | – | 57.9 | 57.1 |
| VideoPrism-g | 39.7 | 71.0 | 58.1 | 83.3 | – | – |
| SigLIP2-g-opt | 43.1 | 34.2 | 55.8 | 74.6 | – | – |
| PE-Core-G Image | 44.3 | 35.2 | 54.3 | 73.9 | – | – |
| ViCLIP | 42.4 | 41.3 | 49.1 | 75.1 | 31.5 | 31.5 |
| **ViCLIP + KeyScore** | **51.3** | **49.8** | **57.9** | **83.4** | **41.2** | **40.9** |
| PE-Core-G Video | 51.2 | 49.9 | 59.7 | 85.4 | 43.1 | 45.1 |
| **PE-Core-G Video + KeyScore** | **63.9** | **60.5** | **89.2** | **89.2** | **60.4** | **60.3** |

KeyScore improves **two different frozen backbones** (+9~10 R@1 T2V on ViCLIP,
+12.7 on PE-Core), indicating the selector is backbone-agnostic rather than tuned to
one encoder.

## 3. Video question answering — Video-MME (no subtitles, Qwen2-VL 7B)

| Method | #Frames | Overall | Short | Medium | Long |
| --- | --- | --- | --- | --- | --- |
| Qwen2-VL | 8 | 53.0 | 64.1 | 49.3 | 45.6 |
| + Q-Frame | 4+8+32 | 58.3 | **69.4** | 57.1 | 48.3 |
| + AKS | 32 | **59.9** | – | – | – |
| **+ KeyScore (ours)** | **8** | 59.4 | 68.2 | **58.5** | **51.7** |

- +6.4 overall over the 8-frame baseline at the **same** frame budget.
- Gains grow with video length (long: 45.6 -> 51.7), consistent with redundancy
  being the limiting factor.
- With 8 frames KeyScore closes most of the gap to the 32-frame AKS result and
  surpasses Q-Frame's multi-resolution result.

## 4. Runtime and compute fairness (MSR-VTT, 1k videos, A100)

Per-video wall-clock in seconds.

| Method | Frames | Sampling | Embedding | Scoring | Total | R@1 (T2V) |
| --- | --- | --- | --- | --- | --- | --- |
| UFP-8 | 8.00 | 0.085 | 1.476 | – | 1.561 | 50.3 |
| UFP (compute-matched) | 7.70 | 0.072 | 1.263 | – | 1.335 | 50.4 |
| UFP (frame-matched) | 4.27 | 0.042 | 0.749 | – | 0.791 | 47.0 |
| **STACFP + KeyScore** | **4.27** | 0.650 | 1.263 | 0.0003 | 1.914 | **59.8** |

- *Compute-matched* UFP encodes the same number of frames as the STACFP candidate
  pool; *frame-matched* UFP uses KeyScore's final frame count.
- Uniform accuracy is essentially flat in frame count (8 vs 7.7 frames: 50.3 vs
  50.4), so a runtime-matched UFP cannot close the ~9-point gap.
- Scoring costs **0.3 ms per video** (<0.02% of total) because leave-one-out drop
  scores have a closed form under mean pooling and reuse cached embeddings.

Raw data: [`../results/msrvtt_runtime_1k.json`](../results/msrvtt_runtime_1k.json).

## 5. Controlled query-aware comparison

Every method below shares the **same frozen PE backbone** and the **same STACFP
candidate frames**; `QueryTopK` is additionally frame-matched to KeyScore. Only the
selection rule differs. 1000 MSR-VTT test videos.

| Method | ASF | T2V R@1 | R@5 | R@10 | V2T R@1 |
| --- | --- | --- | --- | --- | --- |
| STACFP-mean (no selection) | 7.70 | 49.0 | 73.8 | 81.3 | 49.3 |
| UFP-8 | 8.00 | 50.3 | 74.1 | 81.8 | 48.9 |
| QueryTopK (frame-matched) | 4.27 | 59.7 | 80.2 | 88.9 | 58.7 |
| **KeyScore (S+T+D)** | **4.27** | **59.8** | 80.2 | 88.9 | 58.7 |

Interpretation: the ~10-point jump comes from **selecting** a query-conditioned
subset, not from the proposal stage and not from extra frames. On this aggregate
retrieval metric KeyScore is on par with a strong relevance-only top-K baseline at
equal frame count; the components beyond semantic relevance matter more on
multi-moment and long-video settings (see §1 DiDeMo and §3 Video-MME long).

Raw data: [`../results/msrvtt_query_aware.json`](../results/msrvtt_query_aware.json).

## 6. STACFP timestamp-weight sensitivity

`gamma_time` scales the normalized timestamp feature appended to the colour
histogram. 300 MSR-VTT videos.

| `gamma_time` | ASF | T2V R@1 |
| --- | --- | --- |
| 0.0 | 5.44 | 72.0 |
| 1.0 | 4.94 | 71.3 |
| 2.0 | 4.80 | 72.0 |
| 5.0 | 4.40 | 72.3 |
| 10.0 | 4.00 | 72.3 |
| 20.0 | 3.54 | **72.7** |

Accuracy stays within 1.4 points across a 20x range while the frame count drops 35%
(5.44 -> 3.54). The timestamp term therefore buys compression essentially for free,
and the method is not sensitive to this hyperparameter.

Raw data: [`../results/msrvtt_gamma_time_sweep.json`](../results/msrvtt_gamma_time_sweep.json).

## 7. MLVU (Qwen2-VL 7B, 8 frames from 32 candidates)

1018 questions.

| Selector | Overall | 2_needle | 6_anomaly_reco |
| --- | --- | --- | --- |
| Uniform | 49.3 | 60.9 | **66.5** |
| Relevance-only top-K | **57.4** | 73.9 | 60.5 |
| KeyScore | 57.1 | **74.2** | – |

This is a deliberately reported negative/neutral result: on MLVU overall, KeyScore is
**not** better than a relevance-only top-K selector (57.1 vs 57.4). Its advantage is
concentrated on needle-in-a-haystack retrieval (74.2 vs 60.9 for uniform), while
uniform sampling remains strongest on anomaly recognition, where broad temporal
coverage matters more than query relevance.

Raw data: [`../results/mlvu_summary.json`](../results/mlvu_summary.json).

## 8. Zero-shot action classification — HMDB-51

| Model | Resolution | F1 | FRR-Avg |
| --- | --- | --- | --- |
| InternVL | 224 | 0.555 | 0.915 |
| InternVideo2 | 224 | 0.539 | 0.915 |
| SigLIP2-g-opt | 384 | 0.518 | 0.915 |
| **PE-Core-G Video + KeyScore** | 448 | **0.675** | **0.972** |

Caveats: the selection threshold is swept and the best operating point is reported.
Because that threshold is chosen on the evaluation set rather than a held-out split,
the F1 above is an **optimistic upper bound**. Baselines also use different encoders
and input resolutions, so this is an auxiliary cross-encoder generality check rather
than a controlled same-backbone ablation.

---

## Known limitations

- Retrieval re-selects frames **per query–video pair**. Candidate embeddings are
  cached and reused, so the marginal cost per query is a text encode plus ~0.3 ms of
  scoring, but the selection itself is not shared across queries.
- STACFP sampling (0.65 s/video) dominates KeyScore's overhead; the scoring stage
  itself is effectively free.
- KeyScore does not uniformly dominate relevance-only top-K. See §7.
- Gains are largest under high visual redundancy; on short, low-redundancy clips
  uniform sampling is already close to optimal.
