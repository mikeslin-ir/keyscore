# MAVU 2027 resubmission experiments

## Scope and safeguards

- Submission: MAVU at WACV 2027, OpenReview paper 8.
- Deadline published by the workshop: October 20, 2026, 11:59 PM PST.
- Content baseline: the existing NeurIPS main and supplementary PDFs.
- Compute: one currently idle NVIDIA GB10, explicitly authorized by the author.
- The GPU became occupied after the initial inventory. The author subsequently
  authorized sharing it for the 32-video pilot, without concurrent QA-model
  launches by this task. Timings from this run are exploratory, not publication-
  ready isolated-GPU latency measurements.
- Preserve legacy outputs. Write new runs below `outputs/mavu2027/`.
- Do not promote pilot results, oracle-conditioned diagnostics, or test-set
  maxima into paper claims.
- Existing headline numbers remain unverified until their originating
  configurations and retrieval protocols are traced.

## P0: protocol-correct retrieval

For every text query q and every candidate video V, select frames from V
conditioned on q, then score that pair. Never select a database video's
representation using its matched benchmark caption and reuse it for other
queries. If retained, that old protocol must be labeled an oracle diagnostic.

Initial runs use the frozen PE-Core-bigG/14-448 encoder, STACFP candidates,
and the first caption per video, explicitly labeled a single-caption
protocol rather than assumed identical to every published benchmark.
Record the ordered video list, caption rule, annotation hash, backbone,
proposal settings, effective budgets, software versions, and source hashes.

1. Unit tests: batched scoring versus the original scorer; query permutation;
   identical-video non-separability; selector budgets; greedy and MMR behavior.
2. Deterministic pilot: first 32 videos of the MSR-VTT Miech 1k list, K=4.
   Check real embeddings, all query-video pairs, finite scores, timing, and
   the difference from explicitly labeled own-caption conditioning.
3. Scale to the complete Miech 1k list at K=2,4,8 after the pilot passes.
   All selectors must share candidate embeddings and effective frame budgets.
4. Before making cross-paper comparisons, match the official query set,
   split, multiple-positive handling, and backbone evaluation protocol.

Initial pilot command, from the repository root:

```sh
export KEYSCORE_DATA_ROOT=/path/to/datasets
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
python scripts/eval_pairwise_retrieval.py \
  --dataset-root "$KEYSCORE_DATA_ROOT/MSRVTT" \
  --list "$KEYSCORE_DATA_ROOT/MSRVTT/structured-symlinks/test_list_miech.txt" \
  --cache outputs/mavu2027/pe_msrvtt_cache \
  --out outputs/mavu2027/msrvtt-pilot32-k4 \
  --limit 32 --k 4
```

After validating the pilot, omit `--limit` and choose a new output directory
to evaluate the complete list. Reuse the cache for each prespecified K.
Each run writes `metadata.json`, `summary.json`, and `per_pair.pt`, including
source/input hashes, complete score matrices, selected indices, timings,
and the explicitly separate own-caption-conditioned diagnostic.

Selectors:

- Mean pooling of all candidates (query-independent reference).
- Query-TopK / semantic-only.
- S+D with fixed weights (0.8, 0, 0.2).
- Full KeyScore with fixed weights (0.7, 0.1, 0.2).
- Greedy forward selection maximizing cosine of the pooled selected subset.
- MMR with prespecified relevance weight 0.7.

These are prespecified diagnostic settings, not newly validation-optimized
settings. Do not tune them on the test results. Any future tuning must use a
disjoint validation split, record the complete search, and freeze the chosen
configuration before test evaluation.

Report T2V/V2T R@1/5/10, selected-frame redundancy, candidate redundancy,
paired per-query outcomes, and selector online wall-clock. Separate offline
proposal/encoding from online database-wide selection, aggregation, and
scoring. Cached input does not make pairwise online work disappear.

## P1: scoring value, redundancy, and Video QA

1. DiDeMo: repeat the same-candidate, same-K selector comparison. Test whether
   full-minus-semantic gains increase with measured candidate redundancy;
   do not use duration alone as a proxy.
2. Video-MME: use one fixed candidate cache and Qwen2-VL 7B, no subtitles,
   K=8. Run uniform, semantic-only, S+D, full, greedy, and MMR with the same
   question prompt, generation settings, image preprocessing, and frame order.
3. Preserve per-question predictions and selected indices; analyze overall
   and per-length accuracy, candidate/selected redundancy, and video-clustered
   paired uncertainty. Existing scored-versus-uniform QA predictions cannot
   stand in for a controlled full-versus-semantic comparison.
4. Report the complete selector-plus-QA cost, including the selection encoder,
   not only the frames passed to Qwen.

The retrieval protocol check is the first run in this work session. Video-MME
inference and parameter sweeps are later stages, not implicitly running.

## Completed initial pilot

The first 32 listed MSR-VTT videos were evaluated against all 32 queries
(1,024 query-video pairs), with fresh frozen-PE embeddings and K=4.
Results are stored in `outputs/mavu2027/msrvtt-pilot32-k4/summary.json`;
full pair matrices and selected indices are in the adjacent `per_pair.pt`.

| Selector | T2V R@1 | V2T R@1 |
| --- | ---: | ---: |
| All-candidate mean | 87.500 | 81.250 |
| Semantic Top-K | 84.375 | 87.500 |
| S+D | 84.375 | 87.500 |
| Full KeyScore | 84.375 | 87.500 |
| Greedy subset cosine | 87.500 | 84.375 |
| MMR | 87.500 | 84.375 |

The isolated own-caption-conditioned full-score diagnostic gives 87.500 /
90.625, versus the proper full-score 84.375 / 87.500. The diagnostic is not
a valid standard retrieval result and is not included as a benchmark arm.

All stored matrices were checked to be finite and 32x32. Nine real pair
scores were independently compared with the original single-query scorer
and matched within floating-point tolerance. All 24 targeted unit tests pass.

This is a small 32-video database, not the 1k benchmark. The pilot does not
show a full-score advantage over semantic-only and cannot establish
equivalence or justify general accuracy claims. The paired bootstrap interval
is degenerate because both selectors have identical R@1 correctness on every
pilot query; this is not proof that the population difference is zero.

## Completed full-list run

After the pilot and correctness checks passed, the complete 1,000-video
Miech list completed at the same prespecified K=4 on the GB10, reusing the
pilot caches and preparing the remaining videos.

- Log: `outputs/mavu2027/msrvtt-1k-k4.log`
- Output directory: `outputs/mavu2027/msrvtt-1k-k4/`
- Completed outputs: `summary.json`, `metadata.json`, and `per_pair.pt`.
- Descriptive manuscript analysis: `manuscript_analysis.json`, derived from
  the saved matrices and selected indices without a model rerun.
- No Video-MME/Qwen inference or parameter tuning is running.

| Selector | T2V R@1 | V2T R@1 | Selected redundancy |
| --- | ---: | ---: | ---: |
| All-candidate mean | 49.0 | 49.3 | 0.824 |
| Semantic Top-K | 50.7 | 50.9 | 0.832 |
| S+D | 50.5 | 50.8 | 0.829 |
| Full KeyScore | 50.5 | 50.8 | 0.833 |
| Greedy subset cosine | 50.4 | 50.8 | 0.815 |
| MMR | 50.6 | 50.5 | 0.785 |

All six persisted score matrices were verified finite and 1000x1000.
Full-minus-semantic T2V R@1 is -0.2 percentage points, with paired-bootstrap
95% interval [-0.7, 0.2]. Two queries favor full scoring and four favor
semantic-only (exact McNemar p=0.6875).
The full and semantic selectors have identical selected sets for 92.5% of
pairs and mean selected-index overlap of 98.1%.
Low/middle/high candidate-redundancy strata do not show an increasing
full-minus-semantic gain. These are descriptive post-hoc analyses, not
tuning or causal evidence.

The isolated own-caption-conditioned diagnostic obtains 58.7/55.9, versus
proper pairwise full scoring at 50.5/50.8. It remains explicitly excluded
from benchmark accuracy claims.

## P2: manuscript integration

- Completed: the decisive same-budget semantic-versus-full results are in
  the main paper, along with MLVU, redundancy strata, and greedy/MMR results.
- Completed: candidate-context similarity is described as representativeness, not as an
  independent temporal or guaranteed diversity term.
- Completed: the main distinguishes full-candidate leave-one-out utility from
  selected-subset value and contains both stage algorithms and a counterexample.
- Link each table row to an exact run/configuration and its selection split.
- Do not claim a speedup from final selected-frame count alone.
- Completed with author approval: withdrawn unverified retrieval headlines and
  conflicting accuracy/efficiency claims from the WACV version.
- The available Video-MME 1,023-question eight-frame checkpoint does not
  reproduce the legacy headline: scored 576/1,023, uniform 578/1,023.
  Do not promote its old overall or per-length table without a traceable run.
- The WACV supplement now contains only technical details supporting the
  main-paper settings; legacy additional-dataset tables are excluded.
- Remaining: other budgets, validation-selected configurations, and controlled
  Video-MME comparisons are not completed by the current 1k K=4 run.
