"""Run from the checkout with: python -m examples.score_frames."""

import torch

from keyscore import KeyScore
from keyscore.scoring import METHODS


def main() -> None:
    generator = torch.Generator().manual_seed(42)
    frames = torch.randn(8, 16, generator=generator)
    query = torch.randn(16, generator=generator)
    selector = KeyScore()
    components = selector.score(frames, query)
    print("Synthetic embeddings; these are not benchmark results.")
    for name, values in components.items():
        print(f"{name}: {values.tolist()}")
    for method in METHODS:
        result = selector.select(frames, query, k=4, method=method)
        chronological = result.indices.sort().values.tolist()
        print(f"{method}: temporal indices={chronological}, cosine={result.similarities.item():.4f}")


if __name__ == "__main__":
    main()
