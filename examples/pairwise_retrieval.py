"""Run from the checkout with: python -m examples.pairwise_retrieval."""

import torch

from keyscore import KeyScore


def score_database(
    candidate_pools: list[torch.Tensor], queries: torch.Tensor, k: int = 4
) -> torch.Tensor:
    if not candidate_pools:
        raise ValueError("The video database must not be empty.")
    selector = KeyScore()
    return torch.stack([
        selector.select(frames, queries, k=k).similarities
        for frames in candidate_pools
    ], dim=1)


def main() -> None:
    generator = torch.Generator().manual_seed(42)
    videos = [torch.randn(n, 16, generator=generator) for n in (6, 8, 10)]
    queries = torch.randn(2, 16, generator=generator)
    matrix = score_database(videos, queries)
    print("Synthetic embeddings; each query selects from every video independently.")
    print("Query x video shape:", tuple(matrix.shape))
    print(matrix)
    print("Highest-scoring video per query:", matrix.argmax(dim=1).tolist())


if __name__ == "__main__":
    main()
