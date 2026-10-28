"""Pure arithmetic for retrieval metrics. No chunk/span knowledge here.

Every function takes only sequences of numbers so every value is
hand-checkable (docs/design/phase-3.md §6.1). Metrics never re-sort their
inputs: they consume the retriever's rank order, and ties are broken upstream
by the vector store's own row order (Tradeoffs #5) -- re-sorting inside a
metric would hide a retriever's own nondeterminism.
"""

import math
from collections.abc import Sequence


def dcg(gains: Sequence[float]) -> float:
    """sum(g_i / log2(i + 2)) over 0-based ranks -- the standard discount."""
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def recall_at_k(first_covered_ranks: Sequence[int | None], k: int) -> float:
    """Fraction of relevant spans first covered at a rank < k ("span recall", Tradeoffs #2).

    Raises:
        ValueError: on an empty sequence (recall is undefined with no
            relevant spans) or `k < 1`.
    """
    if not first_covered_ranks:
        raise ValueError("first_covered_ranks must not be empty")
    if k < 1:
        raise ValueError("k must be >= 1")
    covered = sum(1 for rank in first_covered_ranks if rank is not None and rank < k)
    return covered / len(first_covered_ranks)


def reciprocal_rank(gains: Sequence[int], k: int, threshold: int = 1) -> float:
    """1/(r+1) for the first 0-based rank r < k with gains[r] >= threshold; 0.0
    if there is none (Tradeoffs #3).

    Raises:
        ValueError: on `k < 1`.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    for rank, gain in enumerate(gains[:k]):
        if gain >= threshold:
            return 1.0 / (rank + 1)
    return 0.0


def ndcg_at_k(gains: Sequence[int], ideal_gains: Sequence[int], k: int) -> float:
    """dcg(gains[:k]) / dcg(sorted(ideal_gains, reverse=True)[:k]).

    Neither argument is re-sorted: `gains` is the retriever's order and ties
    are the retriever's to break (Tradeoffs #5); `ideal_gains` is consumed via
    `sorted()`, which returns a new list rather than mutating the argument.

    Raises:
        ValueError: if `ideal_gains` is empty or sums to 0, or `k < 1`.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not ideal_gains or sum(ideal_gains) == 0:
        raise ValueError("ideal_gains must be non-empty and sum to more than 0")
    ideal_sorted = sorted(ideal_gains, reverse=True)
    idcg = dcg(ideal_sorted[:k])
    return dcg(gains[:k]) / idcg


def mean_or_none(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or None for an empty sequence -- an unmeasured metric is never 0.0."""
    if not values:
        return None
    return sum(values) / len(values)
