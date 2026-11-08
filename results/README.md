# Result provenance

## Current revised-manuscript evidence

| File | Provenance | Scope |
| --- | --- | --- |
| [msrvtt_pairwise_k4.json](msrvtt_pairwise_k4.json) | Compact export of the completed fixed-K run and saved-pair analysis | 1k videos, one caption per video, 1M pairs, six methods |
| [mlvu_summary.json](mlvu_summary.json) | Recomputed from 1,018 saved paired predictions; not new inference | Counts, all four tasks, exact question-level McNemar tests |
| [mlvu_question_ids.json](mlvu_question_ids.json) | Ordered task/video/question IDs from those predictions | Exact slice for the reproduction CLI; no videos or question text |

The compact retrieval export preserves measured metrics, configuration, hashes,
timing scope, and descriptive analysis. It omits machine-specific absolute paths,
process IDs, cached tensors, and the 1,000 annotation captions. Its original source
hashes identify the measured run, not a new benchmark of the refactored release.
New experiments write full provenance and matrices under ignored `outputs/`.

## Historical artifacts: not current benchmark claims

| File | Original script | Limitation |
| --- | --- | --- |
| [msrvtt_query_aware.json](msrvtt_query_aware.json) | `eval_query_aware_baseline.py` | Earlier conditioning/threshold protocol; not the current pairwise benchmark |
| [msrvtt_gamma_time_sweep.json](msrvtt_gamma_time_sweep.json) | `eval_gamma_time_sweep.py` | Earlier 300-video sweep; not a prespecified full-list validation study |
| [msrvtt_runtime_1k.json](msrvtt_runtime_1k.json) | `benchmark_msrvtt_mini_runtime.py` | Historical A100 timing/accuracy setup; not the GB10 pairwise run |

These files are retained for traceability, not used to support current accuracy,
diversity, transfer, or latency claims. Their numbers must not be substituted for
the revised results.

See [RESULTS.md](../docs/RESULTS.md) for interpreted tables and
[REPRODUCE.md](../docs/REPRODUCE.md) for commands.
