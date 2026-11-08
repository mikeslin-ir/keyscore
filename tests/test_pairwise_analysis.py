import pytest
import torch

from keyscore.evaluation import paired_mcnemar
from scripts.analyze_pairwise_retrieval import analyze


def test_paired_comparison_matches_manuscript_discordants():
    result = paired_mcnemar([1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1])
    assert result == {"a_only_correct": 2, "b_only_correct": 4, "two_sided_exact_p": 0.6875}
    assert paired_mcnemar([1], [1])["two_sided_exact_p"] == 1.0


@pytest.mark.parametrize("a,b", [([], []), ([1], []), ([2], [0])])
def test_invalid_paired_outcomes(a, b):
    with pytest.raises(ValueError):
        paired_mcnemar(a, b)


def test_strata_keep_full_database_and_compare_selected_sets():
    semantic = torch.eye(3)
    full = semantic.clone()
    full[0] = torch.tensor([0.0, 1.0, 0.0])
    indices = [torch.zeros(3, 1, dtype=torch.long) for _ in range(3)]
    result = analyze(
        {"candidate_pairwise_cosine_by_video": [0.9, 0.2, 0.5]},
        {"score_matrices": {"semantic": semantic, "keyscore": full},
         "semantic": {"selected_indices_by_video": indices},
         "keyscore": {"selected_indices_by_video": indices}},
    )
    groups = result["candidate_redundancy_strata_by_target_video"]
    assert [s["n"] for s in groups] == [1, 1, 1]
    assert groups[0]["redundancy_min"] == 0.2
    assert groups[2]["t2v_r1"]["keyscore"] == 0.0
    assert result["identical_selected_set_fraction"] == 1.0
    assert result["mean_semantic_full_selected_overlap_fraction"] == 1.0
    assert result["full_vs_semantic_mcnemar"]["semantic_only_correct"] == 1
