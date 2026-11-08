import pytest
import torch
import torch.nn.functional as F

from keyscore.scoring.frame_scorer import FrameScorer, MeanPooling, TextGuidedAttentionPooling
from keyscore.scoring.selection import METHODS, pooled_pairs, select_indices
from scripts.eval_pairwise_retrieval import evaluate, recall


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("count", [1, 3, 12])
def test_batched_scores_match_original_scorer(count, normalize):
    generator = torch.Generator().manual_seed(123)
    frames = torch.randn(count, 8, generator=generator, dtype=torch.float64)
    queries = torch.randn(5, 8, generator=generator, dtype=torch.float64)
    scorer = FrameScorer(MeanPooling(), 0.7, 0.1, 0.2, normalize_each=normalize)
    batched = scorer.compute_queries(frames, queries)
    for name in ("scores", "sim_semantic", "sim_temporal", "sim_drop"):
        expected = torch.stack([scorer.compute(frames, query)[name] for query in queries])
        torch.testing.assert_close(batched[name], expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_queries_select_independently_and_preserve_budgets(method):
    generator = torch.Generator().manual_seed(7)
    frames = torch.randn(7, 8, generator=generator, dtype=torch.float64)
    queries = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    indices = select_indices(frames, queries, 4, method)
    assert indices.shape == (4, 4)
    assert all(len(set(row.tolist())) == 4 for row in indices)
    for i, query in enumerate(queries):
        single = select_indices(frames, query[None], 4, method)
        torch.testing.assert_close(indices[i], single[0])
    permutation = torch.tensor([2, 0, 3, 1])
    reordered = select_indices(frames, queries[permutation], 4, method)
    torch.testing.assert_close(reordered, indices[permutation])
    assert select_indices(frames[:2], queries, 4, method).shape == (4, 2)


@pytest.mark.parametrize("method", METHODS)
def test_identical_videos_cannot_be_separated_by_matched_captions(method):
    frames = torch.eye(2, dtype=torch.float64)
    queries = frames.clone()
    selected = select_indices(frames, queries, 1, method)
    proper, pooled = pooled_pairs(frames, queries, selected)
    pairwise_matrix = torch.stack([proper, proper], dim=1)
    torch.testing.assert_close(pairwise_matrix, torch.ones(2, 2, dtype=torch.float64))
    oracle_matrix = queries @ pooled.T
    torch.testing.assert_close(oracle_matrix, torch.eye(2, dtype=torch.float64))


def test_greedy_can_improve_over_duplicate_independent_selection():
    angles = torch.deg2rad(torch.tensor([20.0, 20.0, -40.0], dtype=torch.float64))
    frames = torch.stack([angles.cos(), angles.sin()], dim=1)
    queries = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    independent = select_indices(frames, queries, 2, "keyscore")
    greedy = select_indices(frames, queries, 2, "greedy")
    assert independent.tolist() == [[0, 1]]
    assert set(greedy[0].tolist()) == {0, 2}
    independent_score, _ = pooled_pairs(frames, queries, independent)
    greedy_score, _ = pooled_pairs(frames, queries, greedy)
    assert greedy_score.item() > independent_score.item()


def test_mean_pooling_pair_scores_match_direct_computation():
    frames = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    queries = frames.clone()
    indices = torch.tensor([[0, 1], [1, 0]])
    scores, _ = pooled_pairs(frames, queries, indices)
    expected = F.cosine_similarity(frames[indices].mean(dim=1), queries)
    torch.testing.assert_close(scores, expected)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_budget_is_explicit(k):
    with pytest.raises(ValueError, match="positive integer"):
        select_indices(torch.eye(2), torch.eye(2), k, "semantic")


def test_invalid_embeddings_and_pooling_are_explicit():
    with pytest.raises(ValueError, match="finite"):
        select_indices(torch.tensor([[float("nan"), 0.0]]), torch.eye(2), 1, "semantic")
    with pytest.raises(ValueError, match="Zero-norm"):
        select_indices(torch.zeros(1, 2), torch.eye(2), 1, "semantic")
    scorer = FrameScorer(TextGuidedAttentionPooling())
    with pytest.raises(ValueError, match="MeanPooling"):
        scorer.compute_queries(torch.eye(2), torch.eye(2))


def test_evaluator_keeps_oracle_diagnostic_separate():
    records = [
        {"frames": torch.eye(2), "text": query}
        for query in torch.eye(2)
    ]
    summary, details = evaluate(records, torch.device("cpu"), 1)
    for matrix in details["score_matrices"].values():
        torch.testing.assert_close(matrix[:, 0], matrix[:, 1])
        assert recall(matrix)["1"] == 50.0
    diagnostic = summary["diagnostics"]["own_caption_conditioned_keyscore"]
    assert diagnostic["valid_standard_retrieval_result"] is False
    assert diagnostic["recall_t2v"]["1"] == 100.0
