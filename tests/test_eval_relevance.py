"""Acceptance criterion §6.4: credit_hits hand-built scenarios."""

import pytest

from docsense.eval.relevance import credit_hits, relevant_spans, spans_overlap
from docsense.eval.schema import GoldExample, LabeledSpan
from docsense.retrieve.schema import RetrievedChunk


def _hit(rank: int, arxiv_id: str, page_start: int, page_end: int, score: float = 1.0):
    return RetrievedChunk(
        chunk_id=f"{arxiv_id}_chunk_{rank:04d}",
        arxiv_id=arxiv_id,
        page_start=page_start,
        page_end=page_end,
        text="irrelevant text",
        score=score,
        rank=rank,
    )


def _span(arxiv_id: str, page_start: int, page_end: int, relevance: int):
    return LabeledSpan(
        arxiv_id=arxiv_id,
        page_start=page_start,
        page_end=page_end,
        relevance=relevance,
        note=None,
        source_chunk_id=None,
    )


def test_spans_overlap_boundary_touching_pages():
    assert spans_overlap(4, 5, 5, 6)
    assert spans_overlap(5, 6, 4, 5)


def test_spans_overlap_disjoint():
    assert not spans_overlap(1, 2, 3, 4)


def test_credits_hit_whose_pages_overlap_a_span():
    hits = [_hit(0, "2301.00001", 5, 5)]
    spans = [_span("2301.00001", 4, 5, 2)]
    credited = credit_hits(hits, spans, threshold=1)
    assert credited.gains == [2]
    assert credited.first_covered_ranks == [0]


def test_does_not_credit_different_arxiv_id_with_same_pages():
    hits = [_hit(0, "2301.99999", 5, 5)]
    spans = [_span("2301.00001", 4, 5, 2)]
    credited = credit_hits(hits, spans, threshold=1)
    assert credited.gains == [0]
    assert credited.first_covered_ranks == [None]


def test_second_hit_covering_already_credited_span_gets_gain_zero():
    hits = [_hit(0, "2301.00001", 5, 5), _hit(1, "2301.00001", 5, 5)]
    spans = [_span("2301.00001", 5, 5, 2)]
    credited = credit_hits(hits, spans, threshold=1)
    assert credited.gains == [2, 0]
    assert credited.first_covered_ranks == [0]


def test_hit_covering_two_uncovered_spans_gets_max_relevance_and_credits_both():
    hits = [_hit(0, "2301.00001", 1, 10)]
    spans = [_span("2301.00001", 2, 3, 1), _span("2301.00001", 5, 6, 2)]
    credited = credit_hits(hits, spans, threshold=1)
    assert credited.gains == [2]
    assert credited.first_covered_ranks == [0, 0]


def test_relevance_zero_spans_are_ignored():
    hits = [_hit(0, "2301.00001", 5, 5)]
    spans = [_span("2301.00001", 5, 5, 0)]
    credited = credit_hits(hits, spans, threshold=1)
    assert credited.gains == [0]
    assert credited.first_covered_ranks == []
    assert credited.ideal_gains == []


def test_ideal_gains_are_descending():
    hits = [_hit(0, "2301.00001", 1, 1)]
    spans = [
        _span("2301.00001", 1, 1, 1),
        _span("2301.00001", 2, 2, 2),
        _span("2301.00001", 3, 3, 1),
    ]
    credited = credit_hits(hits, spans, threshold=1)
    assert credited.ideal_gains == [2, 1, 1]


def test_non_contiguous_ranks_raise_value_error():
    hits = [_hit(0, "2301.00001", 1, 1), _hit(2, "2301.00001", 2, 2)]
    spans = [_span("2301.00001", 1, 1, 1)]
    with pytest.raises(ValueError):
        credit_hits(hits, spans, threshold=1)


def test_relevant_spans_filters_by_threshold_and_preserves_order():
    example = GoldExample(
        query_id="q1",
        query="what is x",
        query_type="factoid",
        difficulty="easy",
        unanswerable=False,
        spans=[
            _span("2301.00001", 1, 1, 0),
            _span("2301.00001", 2, 2, 2),
            _span("2301.00001", 3, 3, 1),
        ],
        tags=[],
        labeled_by="tester",
        labeled_at="2024-01-01T00:00:00+00:00",
        pooled_from=["dense"],
        pool_k=20,
    )
    result = relevant_spans(example, threshold=1)
    assert [s.page_start for s in result] == [2, 3]
