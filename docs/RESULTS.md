# Current manuscript results

This document supersedes the older threshold-based retrieval, cross-backbone,
Video-MME, and auxiliary action/summarization headlines. Those are not validated
results for the revised protocol. Original JSONs remain explicitly historical;
see [result provenance](../results/README.md).

## 1. Pairwise MSR-VTT retrieval

Frozen PE-Core-bigG/14 at 448 pixels, Miech 1k video list, first annotation caption
per video, 1,000 queries against 1,000 videos. STACFP produces 7.703 candidates
on average (range 6--29); all selectors retain K=4. The all-candidate mean is
query-independent and is not a frame-matched selector.

Each query selects frames independently from every candidate video. It never
uses another video's matched test caption to build that video's representation.
The same matrix, transposed, gives V2T scores.

| Selector | T2V R@1 | R@5 | R@10 | V2T R@1 | R@5 | R@10 | Selected cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All-candidate mean | 49.0 | 73.8 | 81.3 | 49.3 | 71.4 | 79.1 | 0.824 |
| Semantic / Query-TopK | 50.7 | 73.0 | 81.7 | 50.9 | 72.9 | 82.0 | 0.832 |
| S+D | 50.5 | 73.1 | 81.4 | 50.8 | 72.9 | 81.7 | 0.829 |
| Full KeyScore | 50.5 | 73.0 | 81.8 | 50.8 | 73.0 | 82.1 | 0.833 |
| Greedy subset cosine | 50.4 | 73.3 | 81.8 | 50.8 | 73.1 | 81.9 | 0.815 |
| MMR | 50.6 | 73.5 | 81.6 | 50.5 | 72.7 | 80.1 | 0.785 |

Accuracy is percent. Selected cosine is mean off-diagonal pairwise cosine of unit
frame embeddings across query-video pairs; lower values indicate more embedding
diversity, not necessarily better temporal reasoning.

**Interpretation:** full scoring improves modestly over all-candidate pooling,
but not over semantic-only selection. This single-caption protocol is not a
substitute for standard all-caption benchmark evaluation.

Full minus semantic T2V R@1 is **-0.2 percentage points**. The paired bootstrap
95% percentile interval is **[-0.7, 0.2]**, using 2,000 resamples and seed 42.
Only six queries differ in R@1 correctness: two favor full, four favor semantic;
the exact two-sided McNemar p-value is **0.6875**. This does not establish
equivalence or an additional benefit from the full mixture.

An isolated matched-caption-conditioned diagnostic gives 58.7/55.9 T2V/V2T R@1.
It is **not a valid standard-retrieval result**. It must not be mixed into
the main table or compared as an independently usable video-only index.

## 2. Redundancy and selected-set overlap

Queries are grouped by the candidate redundancy of their matched target video.
All groups retain the full 1k-video retrieval database.

| Redundancy third | Queries | Semantic T2V R@1 | Full T2V R@1 | Difference |
| --- | ---: | ---: | ---: | ---: |
| Low | 334 | 51.5 | 51.5 | 0.0 |
| Middle | 333 | 52.0 | 51.7 | -0.3 |
| High | 333 | 48.6 | 48.3 | -0.3 |

Semantic and full scoring share **98.1%** of selected indices on average and
choose the same set for **92.5%** of query-video pairs. The descriptive
strata do not show larger gains from full scoring as redundancy increases.
These observations were not used to tune parameters and do not establish causality.

## 3. Controlled MLVU question answering

1,018 paired questions over 794 videos: Needle 353, AnomalyReco 200, Count 206,
Order 259. All methods use 32 uniform candidates, frozen CLIP-ViT-H/14 selector
embeddings, and eight frames in temporal order for Qwen2-VL 7B. The selector
receives the question only; answer choices are part of the QA prompt.

| Selector | Needle | AnomalyReco | Count | Order | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 60.9 | 66.5 | 20.4 | 43.2 | 49.3 |
| Semantic / Query-TopK | 73.9 | 60.5 | 32.5 | 52.1 | 57.4 |
| Full KeyScore | 74.2 | 60.0 | 32.0 | 51.4 | 57.1 |

Correct totals are 502, 584, and 581 respectively. Exact paired McNemar tests:

- Uniform vs semantic: p = 5.24e-9.
- Uniform vs full: p = 2.12e-8.
- Semantic vs full: p = 0.5488; seven discordances favor semantic, four favor full.

Question-level tests do not account for within-video dependence. The results
support query-aware selection in this setting, not superiority of the full score.
Uniform is strongest on AnomalyReco; Count and Order can require multiple moments.
These saved predictions are not fresh Qwen inference after the API refactor.

## 4. Cost accounting

The saved retrieval cache records **601.8 s** of proposal generation and
**1,274.9 s** of frame/text encoding on an NVIDIA GB10.

| Cached online pass over 1,000,000 pairs | Seconds |
| --- | ---: |
| Candidate mean | 0.126 |
| Semantic | 1.352 |
| S+D | 1.095 |
| Full KeyScore | 1.053 |
| Greedy | 1.970 |
| MMR | 1.494 |

These are exploratory, single-run instrumented measurements, with concurrent GPU
activity not excluded. They exclude proposal/encoder work, retrieval ranking,
serialization, and the isolated diagnostic. Their ordering does not prove full
scoring intrinsically faster than semantic-only.

Shared sums make full leave-one-out scoring O(Md) per query-video pair.
Database-wide pairwise selection still costs O(QVMd) for mean pool size M,
before selection/sorting overhead. Cached frame storage is O(VMd), a dense score
matrix is O(QV), and saved selected indices are O(QVK). Selected-frame savings
are not equivalent to encoder or end-to-end savings.

## 5. Reproducibility and evidence boundaries

- Current machine-readable reference: [msrvtt_pairwise_k4.json](../results/msrvtt_pairwise_k4.json).
- MLVU counts/tests: [mlvu_summary.json](../results/mlvu_summary.json).
- Exact MLVU slice: [mlvu_question_ids.json](../results/mlvu_question_ids.json).
- Commands and output schemas: [runbook](REPRODUCE.md).

The earlier Video-MME headline is not reproduced by the available 1,023-question
eight-frame checkpoint (576 scored-correct versus 578 uniform-correct).
That partial checkpoint is not a full-benchmark replacement. Cross-backbone
retrieval tables and threshold-selected action maxima are also excluded from the
current evidence. Additional budgets, validation-selected settings, and a controlled
Video-MME comparison remain future work.
