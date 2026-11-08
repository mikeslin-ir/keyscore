import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from keyscore import KeyScore, SelectionResult
from keyscore.scoring import METHODS, FrameScorer, MeanPooling
from keyscore.scoring.selection import pooled_pairs, select_indices
from examples.pairwise_retrieval import score_database


@pytest.fixture
def embeddings():
    generator = torch.Generator().manual_seed(15)
    return (
        torch.randn(8, 16, generator=generator, dtype=torch.float64),
        torch.randn(3, 16, generator=generator, dtype=torch.float64),
    )


def test_score_components_and_paper_defaults(embeddings):
    frames, queries = embeddings
    selector = KeyScore()
    expected = FrameScorer(MeanPooling(), 0.7, 0.1, 0.2).compute_queries(frames, queries)
    actual = selector.score(frames, queries)
    assert selector.weights == (0.7, 0.1, 0.2)
    for public, legacy in [
        ("scores", "scores"), ("semantic", "sim_semantic"),
        ("representativeness", "sim_temporal"), ("drop_impact", "sim_drop"),
    ]:
        torch.testing.assert_close(actual[public], expected[legacy])
        assert actual[public].shape == (3, 8)
    for q, query in enumerate(queries):
        for name, values in selector.score(frames, query).items():
            torch.testing.assert_close(values, actual[name][q])
    assert FrameScorer(MeanPooling()).alpha == 0.85


@pytest.mark.parametrize("method", METHODS)
def test_selection_matches_existing_helpers_and_shapes(embeddings, method):
    frames, queries = embeddings
    selector = KeyScore()
    result = selector.select(frames, queries, 4, method=method)
    indices = select_indices(frames, queries, 4, method)
    scores, pooled = pooled_pairs(frames, queries, indices)
    assert isinstance(result, SelectionResult)
    torch.testing.assert_close(result.indices, indices)
    torch.testing.assert_close(result.similarities, scores)
    torch.testing.assert_close(result.pooled_embeddings, pooled)
    for q, query in enumerate(queries):
        single = selector.select(frames, query, 4, method=method)
        assert single.indices.shape == (4,)
        assert single.pooled_embeddings.shape == (16,)
        assert single.similarities.ndim == 0
        torch.testing.assert_close(single.indices, indices[q])
        torch.testing.assert_close(single.similarities, scores[q])
    capped = selector.select(frames[:2], queries, 8, method=method)
    assert capped.indices.shape == (3, 2)


def test_custom_weights_are_used_for_scoring_and_selection(embeddings):
    frames, queries = embeddings
    selector = KeyScore(alpha=1.0, beta=0.0, gamma=0.0)
    expected = select_indices(frames, queries, 4, "semantic")
    actual = selector.select(frames, queries, 4)
    torch.testing.assert_close(actual.indices, expected)
    ranked = selector.score(frames, queries)["scores"].argsort(
        dim=1, descending=True, stable=True
    )[:, :4]
    torch.testing.assert_close(actual.indices, ranked)
    semantic_drop = selector.select(frames, queries, method="semantic_drop")
    torch.testing.assert_close(
        semantic_drop.indices, select_indices(frames, queries, 4, "semantic_drop")
    )


@pytest.mark.parametrize("weights", [
    (-0.1, 0.1, 1.0), (math.nan, 0.0, 1.0),
    (math.inf, 0.0, 0.0), (0.7, 0.2, 0.2),
])
def test_invalid_weights_fail_explicitly(weights):
    with pytest.raises(ValueError):
        KeyScore(*weights)


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_mmr_weight_fails_explicitly(value):
    with pytest.raises(ValueError, match="mmr_relevance"):
        KeyScore(mmr_relevance=value)


@pytest.mark.parametrize("operation", ["score", "select"])
def test_public_api_validates_embeddings(operation):
    method = getattr(KeyScore(), operation)
    with pytest.raises(ValueError, match="finite"):
        method(torch.tensor([[math.nan, 0.0]]), torch.ones(2))
    with pytest.raises(ValueError, match="Zero-norm"):
        method(torch.zeros(1, 2), torch.ones(2))
    with pytest.raises(ValueError, match="dtype"):
        method(torch.eye(2), torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="nonempty"):
        method(torch.empty(0, 2), torch.ones(2))
    with pytest.raises(ValueError, match="dimensions"):
        method(torch.eye(2), torch.ones(3))
    with pytest.raises(ValueError, match="text_embs"):
        method(torch.eye(2), torch.ones(1, 1, 2))


def test_constant_components_and_single_candidate():
    selector = KeyScore()
    frames = torch.ones(3, 2)
    query = torch.ones(2)
    scores = selector.score(frames, query)
    assert all(torch.count_nonzero(values) == 0 for values in scores.values())
    assert selector.select(frames, query, 2).indices.tolist() == [0, 1]
    single = selector.select(frames[:1], query, 4)
    assert single.indices.tolist() == [0]
    torch.testing.assert_close(single.similarities, torch.tensor(1.0))


def test_zero_selected_mean_fails_instead_of_returning_a_cosine():
    frames = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    with pytest.raises(ValueError, match="zero-norm mean"):
        KeyScore().select(frames, torch.ones(2), 2, method="semantic")


def test_pairwise_example_keeps_every_query_video_pair_independent(embeddings):
    frames, queries = embeddings
    videos = [frames, frames.clone()]
    matrix = score_database(videos, queries)
    assert matrix.shape == (3, 2)
    torch.testing.assert_close(matrix[:, 0], matrix[:, 1])
    for row, query in enumerate(queries):
        torch.testing.assert_close(
            score_database(videos, query[None])[0], matrix[row]
        )
    with pytest.raises(ValueError, match="empty"):
        score_database([], queries)


def test_selected_cosine_matches_normalized_frame_means(embeddings):
    frames, queries = embeddings
    result = KeyScore().select(frames, queries)
    expected = F.cosine_similarity(
        F.normalize(frames, dim=1)[result.indices].mean(dim=1), queries
    )
    torch.testing.assert_close(result.similarities, expected)


@pytest.mark.parametrize("example", ["score_frames.py", "pairwise_retrieval.py"])
def test_examples_run_without_models_or_datasets(example):
    root = Path(__file__).resolve().parents[1]
    output = subprocess.run(
        [sys.executable, "-m", f"examples.{Path(example).stem}"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert "Synthetic embeddings" in output.stdout
