"""Analyze a saved pairwise run without loading an encoder or rerunning selection."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keyscore.evaluation import paired_mcnemar


def analyze(summary: dict, details: dict) -> dict:
    matrices = details["score_matrices"]
    n = matrices["keyscore"].shape[0]
    redundancy = summary["candidate_pairwise_cosine_by_video"]
    if n < 3 or len(redundancy) != n or any(value is None for value in redundancy):
        raise ValueError("Stratification requires at least three videos with candidate redundancy.")
    for matrix in matrices.values():
        if matrix.shape != (n, n) or not bool(torch.isfinite(matrix).all()):
            raise ValueError("Each score matrix must be finite and square with the same size.")
    candidate_values = torch.tensor(redundancy, dtype=torch.float64)
    if not bool(torch.isfinite(candidate_values).all()):
        raise ValueError("Candidate redundancy must be finite.")
    order = candidate_values.argsort(stable=True)
    hits = {name: matrix.argmax(dim=1) == torch.arange(n) for name, matrix in matrices.items()}
    strata = []
    for label, group in zip(("low", "middle", "high"), torch.tensor_split(order, 3)):
        recalls = {name: float(values[group].sum()) / len(group) * 100
                   for name, values in hits.items()}
        strata.append({
            "stratum": label, "n": len(group),
            "redundancy_min": min(redundancy[i] for i in group.tolist()),
            "redundancy_max": max(redundancy[i] for i in group.tolist()),
            "t2v_r1": recalls,
            "full_minus_semantic_pp": recalls["keyscore"] - recalls["semantic"],
        })
    semantic = details["semantic"]["selected_indices_by_video"]
    full = details["keyscore"]["selected_indices_by_video"]
    if len(semantic) != n or len(full) != n:
        raise ValueError("Selected-index lists must contain one tensor per video.")
    overlap_sum, same_sets = 0.0, 0
    for first, second in zip(semantic, full):
        if first.ndim != 2 or first.shape != second.shape or first.shape[0] != n or first.shape[1] == 0:
            raise ValueError("Paired selectors must have matching nonempty [Q,K] index tensors.")
        overlap = (first.unsqueeze(2) == second.unsqueeze(1)).any(dim=2)
        overlap_sum += float(overlap.sum()) / first.shape[1]
        same_sets += int((first.sort(dim=1).values == second.sort(dim=1).values).all(dim=1).sum())
    comparison = paired_mcnemar(hits["keyscore"].tolist(), hits["semantic"].tolist())
    return {
        "candidate_redundancy_strata_by_target_video": strata,
        "mean_semantic_full_selected_overlap_fraction": overlap_sum / (n * n),
        "identical_selected_set_fraction": same_sets / (n * n),
        "full_vs_semantic_mcnemar": {
            "full_only_correct": comparison["a_only_correct"],
            "semantic_only_correct": comparison["b_only_correct"],
            "two_sided_exact_p": comparison["two_sided_exact_p"],
        },
        "note": "Descriptive post-hoc analysis. All strata retain the complete retrieval database; no tuning.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; choose a new path.")
    summary = json.loads(args.summary.read_text())
    details = torch.load(args.pairs, map_location="cpu", weights_only=True)
    result = analyze(summary, details)
    result["source_summary"] = args.summary.name
    result["source_pairs"] = args.pairs.name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Saved analysis: {args.out}")


if __name__ == "__main__":
    main()
