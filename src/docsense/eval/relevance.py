"""Span matching and credit assignment, separate from metrics.py's pure arithmetic.

`credit_hits` resolves ranked hits against labeled spans with first-credit-only
accounting (Tradeoffs #4): a span pays out exactly once, at the highest rank
that first covers it, so nDCG cannot be inflated by near-duplicate chunks of
the same passage and recall counts spans rather than chunks.
"""

from collections.abc import Sequence

from docsense.eval.schema import CreditedHits, GoldExample, LabeledSpan
from docsense.retrieve.schema import RetrievedChunk


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Inclusive page-interval intersection test."""
    return a_start <= b_end and b_start <= a_end


def relevant_spans(example: GoldExample, threshold: int) -> list[LabeledSpan]:
    """Spans with relevance >= threshold, in file order. Excludes relevance-0 judgments."""
    return [span for span in example.spans if span.relevance >= threshold]


def credit_hits(
    hits: Sequence[RetrievedChunk], spans: Sequence[LabeledSpan], threshold: int
) -> CreditedHits:
    """Resolve ranked hits against labeled spans (Tradeoffs #4, first-credit-only).

    A hit covers a span when `arxiv_id`s match and page intervals overlap.
    Walking ranks in order, each rank's gain is the max relevance of the
    spans it covers that no earlier rank covered (0 if it covers none); every
    span it newly covers records this rank. A span therefore pays out exactly
    once, so nDCG cannot be inflated by near-duplicate chunks of the same
    passage and recall counts spans rather than chunks. Spans with
    `relevance < threshold` (including relevance-0 judgments) are ignored
    entirely -- they never contribute a gain and never appear in
    `ideal_gains` or `first_covered_ranks`.

    Raises:
        ValueError: if `hits` ranks are not `0..n-1`.
    """
    sorted_hits = sorted(hits, key=lambda hit: hit.rank)
    if [hit.rank for hit in sorted_hits] != list(range(len(sorted_hits))):
        raise ValueError(
            f"hit ranks must be exactly 0..{len(sorted_hits) - 1}, "
            f"got {[hit.rank for hit in sorted_hits]}"
        )

    relevant = [span for span in spans if span.relevance >= threshold]
    covered = [False] * len(relevant)
    first_covered_ranks: list[int | None] = [None] * len(relevant)
    gains = [0] * len(sorted_hits)

    for hit in sorted_hits:
        newly_covered_relevances = []
        for idx, span in enumerate(relevant):
            if covered[idx]:
                continue
            if hit.arxiv_id == span.arxiv_id and spans_overlap(
                hit.page_start, hit.page_end, span.page_start, span.page_end
            ):
                covered[idx] = True
                first_covered_ranks[idx] = hit.rank
                newly_covered_relevances.append(span.relevance)
        gains[hit.rank] = max(newly_covered_relevances, default=0)

    ideal_gains = sorted((span.relevance for span in relevant), reverse=True)
    return CreditedHits(
        gains=gains, first_covered_ranks=first_covered_ranks, ideal_gains=ideal_gains
    )
