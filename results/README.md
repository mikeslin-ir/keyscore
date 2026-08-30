# Released results

Raw JSON produced by the scripts in [`../scripts`](../scripts), backing the paper
tables. Each file carries a `metadata` block with the exact hyperparameters used.

| File | Script | Scale | Backs |
| --- | --- | --- | --- |
| `msrvtt_query_aware.json` | `scripts/eval_query_aware_baseline.py` | 1000 MSR-VTT videos | Controlled query-aware selection comparison |
| `msrvtt_gamma_time_sweep.json` | `scripts/eval_gamma_time_sweep.py` | 300 MSR-VTT videos | STACFP `gamma_time` sensitivity |
| `msrvtt_runtime_1k.json` | `scripts/benchmark_msrvtt_mini_runtime.py` | 1000 MSR-VTT videos | Runtime + compute fairness |
| `mlvu_summary.json` | `scripts/videoqa/mlvu_qa_eval.py` | 1018 MLVU questions | Long-video QA, overall + per-task |

Dataset paths in `metadata` are written as `$KEYSCORE_DATA_ROOT/...` placeholders.

Read a summary with:

```bash
python -c "
import json; d = json.load(open('results/msrvtt_query_aware.json'))
for k, v in d.items():
    if k == 'metadata': continue
    print(k, v['avg_selected_frames'], v['retrieval']['recall_t2v'])
"
```

Interpreted tables are in [`../docs/RESULTS.md`](../docs/RESULTS.md).
