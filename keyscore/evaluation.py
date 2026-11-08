"""Paired descriptive statistics shared by the reproduction commands."""

from math import comb
from typing import Sequence, TypedDict


class PairedComparison(TypedDict):
    a_only_correct: int
    b_only_correct: int
    two_sided_exact_p: float


def paired_mcnemar(a: Sequence[int], b: Sequence[int]) -> PairedComparison:
    if not a or len(a) != len(b):
        raise ValueError("Paired outcomes must have equal, nonzero lengths.")
    if any(value not in (0, 1) for values in (a, b) for value in values):
        raise ValueError("Paired outcomes must be zero or one.")
    a_only = sum(bool(x) and not y for x, y in zip(a, b))
    b_only = sum(bool(y) and not x for x, y in zip(a, b))
    n = a_only + b_only
    p = min(1.0, 2 * sum(comb(n, j) for j in range(min(a_only, b_only) + 1)) / 2**n)
    return {"a_only_correct": a_only, "b_only_correct": b_only, "two_sided_exact_p": p}
