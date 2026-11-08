# KeyScore

**Training-Free Query-Conditioned Frame Selection for Video Understanding**

[![arXiv](https://img.shields.io/badge/arXiv-2510.06509-b31b1b.svg)](https://arxiv.org/abs/2510.06509)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

KeyScore selects frames from a candidate pool using frozen, shared-space image/text
embeddings. The public `KeyScore` class exposes semantic relevance, candidate-context
representativeness, and leave-one-out drop impact, plus the five selectors compared
in the revised manuscript. It does not train or load an encoder.

This checkout follows **KeyScore: Training-Free Query-Conditioned Frame Selection
for Video Understanding**, accepted at the MAVU workshop, WACV 2027, not the
earlier threshold-based retrieval headlines or the planned journal extension.
The citation below refers to the existing 2025 preprint until its replacement
is announced. See [current results and limitations](docs/RESULTS.md) and the
[runbook](docs/REPRODUCE.md) for the exact evaluation scope.

<p align="center">
  <img src="assets/pipeline.png" width="90%" alt="Candidate proposals followed by query-conditioned frame scoring"/>
</p>

## Install

Python 3.10+ is required. Install a [PyTorch build appropriate for your
platform](https://pytorch.org/get-started/locally/) first, then:

```bash
git clone https://github.com/mikeslin-ir/keyscore.git
cd keyscore
python -m pip install -e .
```

The core scoring API only requires PyTorch and works on CPU. For video decoding,
STACFP proposals, and pretrained encoders, additionally install:

```bash
python -m pip install -r requirements.txt
```

Model weights download on first encoder use. Datasets and pretrained weights are
not redistributed. The two runnable examples below require neither.

## Usage 1: score and select from embeddings

```python
import torch
from keyscore import KeyScore

# Synthetic inputs for this example. In an application, use image/text embeddings
# from the same frozen backbone, with matching dimension, dtype, and device.
generator = torch.Generator().manual_seed(42)
frames = torch.randn(8, 1280, generator=generator)  # [N,D], chronological input
query = torch.randn(1280, generator=generator)     # [D]

selector = KeyScore()  # alpha=0.7, beta=0.1, gamma=0.2
components = selector.score(frames, query)
print(components["semantic"])
print(components["representativeness"])
print(components["drop_impact"])
print(components["scores"])

selection = selector.select(frames, query, k=4)
print(selection.indices)             # [min(K,N)], in selection order
print(selection.pooled_embeddings)   # [D], normalized selected-frame mean
print(selection.similarities)        # scalar cosine with the query
temporal_indices = selection.indices.sort().values
```

Run the complete example, including all five selectors:

```bash
python -m examples.score_frames
```

### API and shapes

`score(frames, query)` returns four normalized per-candidate tensors.
`select(frames, query, k=4, method="keyscore")` returns a `SelectionResult`.

| Input / output | One query | Multiple queries |
| --- | --- | --- |
| Frame embeddings | `[N,D]` | `[N,D]` |
| Query embeddings | `[D]` | `[Q,D]` |
| Each component and weighted score | `[N]` | `[Q,N]` |
| Selected indices | `[min(K,N)]` | `[Q,min(K,N)]` |
| Pooled embeddings | `[D]` | `[Q,D]` |
| Query-video cosine | scalar | `[Q]` |

Inputs must be finite floating-point tensors with matching device/dtype and
nonzero frame/query norms. A selected subset with zero mean norm is rejected:
its cosine is undefined. Ties use candidate-index order. Sort selected indices
before passing frames to a temporal QA model, assuming the candidates were
originally chronological.

| `method` | Selection rule |
| --- | --- |
| `keyscore` | Weighted score; defaults `(0.7, 0.1, 0.2)` |
| `semantic` | Query-TopK; weights `(1, 0, 0)` |
| `semantic_drop` | S+D; weights `(0.8, 0, 0.2)` |
| `greedy` | Recompute the pooled-subset cosine at each addition |
| `mmr` | Relevance minus maximum selected-frame similarity; relevance weight `0.7` |

For example, `selector.select(frames, query, k=4, method="mmr")` uses MMR.
Custom `KeyScore(alpha=..., beta=..., gamma=...)` weights affect `score()` and
the `keyscore` selector only; named baselines retain their prespecified weights.
`mmr_relevance` is configurable separately. Weights must be finite, nonnegative,
and sum to one. These defaults are prespecified, not claimed validation-optimal.

**Compatibility:** `FrameScorer`, `MeanPooling`, and `TextGuidedAttentionPooling`
remain available from `keyscore.scoring`. Historical `FrameScorer` defaults
remain `(0.85, 0.05, 0.1)`; pass explicit weights or use `KeyScore` for the
current manuscript. The legacy key `sim_temporal` means representativeness,
not a direct temporal or guaranteed diversity measure.

## Usage 2: queries against a video database

Every query must select frames from **every candidate video using that query**.
Do not select each database video once using its own matched evaluation caption.

```python
matrix = torch.stack([
    selector.select(video_frames, query_embeddings, k=4).similarities
    for video_frames in candidate_pools
], dim=1)  # [number of queries, number of videos]
```

Here `candidate_pools` is a list of `[N_j,D]` tensors, and `query_embeddings`
is `[Q,D]`. The runnable example supplies synthetic inputs:

```bash
python -m examples.pairwise_retrieval
```

Embeddings can be cached, but the selected representations depend on the current
query-video pair. This is not a reusable single-vector nearest-neighbor index.

### Connecting a real video

After installing the full requirements, replace synthetic embeddings with:

```python
from keyscore.encoders.perception_encoder import PerceptionEncoder
from keyscore.proposals import FrameSampler_STACFP

sampler = FrameSampler_STACFP(
    frame_step=5, gamma_time=5.0, k_min=6, k_max=30, random_state=42,
)
paths = sampler.extract_frames("video.mp4", "outputs/example/candidates")
encoder = PerceptionEncoder()
frames = encoder.encode_images(paths)
query = encoder.encode_text("a person is playing an acoustic guitar")[0]
selection = selector.select(frames, query, k=4)
selected_paths = [paths[i] for i in selection.indices.sort().values.tolist()]
```

STACFP is query-agnostic histogram/time clustering, not a guarantee of semantic
coverage. Pass these settings explicitly: historical sampler defaults differ.
The MLVU study instead uses 32 **uniform** candidates and selects eight frames.

## Method and evidence

For unit frame embeddings `z_i`, unit query `t`, and normalized candidate mean `v`:

- Semantic relevance: `S_i = cosine(z_i, t)`.
- Representativeness: `R_i = cosine(z_i, v)`.
- Drop impact: `D_i = cosine(v, t) - cosine(mean_without_i, t)`.

Each component is independently min-max normalized over candidates for each
query, with epsilon `1e-8`; ranges at most epsilon map to zero. The weighted
sum ranks candidates. Drop impact is a marginal in the **complete candidate
pool**, not the evolving selected subset, and reuses semantic/geometric evidence.
Independent Top-K therefore does not guarantee optimal subsets or diversity.

Current controlled results:

| Setting | Baseline | Semantic-only | Full KeyScore |
| --- | ---: | ---: | ---: |
| MSR-VTT 1k, T2V R@1, K=4 | 49.0 (all-candidate mean) | 50.7 | 50.5 |
| MSR-VTT 1k, V2T R@1, K=4 | 49.3 (all-candidate mean) | 50.9 | 50.8 |
| MLVU, 1,018 questions, K=8 | 49.3 (uniform) | 57.4 | 57.1 |

The retrieval run uses **one caption per video**, not the standard all-caption
benchmark. The all-candidate reference pools 7.703 frames on average and is not
frame-matched. Neither controlled setting establishes superiority of the full
score over semantic-only selection. Smaller selected-frame budgets also do not
imply fewer proposal encodings or demonstrated end-to-end speedups.

## Runbook and released data

- [Runbook](docs/REPRODUCE.md): installation, smoke tests, pilot/full retrieval,
  saved-run analysis, and the exact MLVU question manifest.
- [Results](docs/RESULTS.md): full tables, uncertainty, redundancy, and timing scope.
- [Result provenance](results/README.md): current summaries versus historical files.
- [Experiment plan](docs/MAVU_EXPERIMENT_PLAN.md): remaining uncompleted comparisons.

Current entry points are `scripts/eval_pairwise_retrieval.py`,
`scripts/analyze_pairwise_retrieval.py`, and `scripts/videoqa/mlvu_qa_eval.py`.
Other retrieval/sweep scripts remain historical; their older headline numbers
are not current protocol-correct results.

## Citation

The citation for the existing preprint is also available in [CITATION.bib](CITATION.bib):

```bibtex
@article{lin2025captions,
  title={From captions to keyframes: Keyscore for multimodal frame scoring and video-language understanding},
  author={Lin, Shih-Yao and Paul, Sibendu and Chen, Caren},
  journal={arXiv preprint arXiv:2510.06509},
  year={2025}
}
```

## License

Source code is released under the [MIT License](LICENSE).
Datasets, pretrained backbones, and third-party dependencies retain their own
licenses and are not redistributed by this repository.
