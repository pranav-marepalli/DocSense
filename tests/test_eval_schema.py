"""Acceptance criteria §6.5-6.6: schema round-trips and validator rejections."""

import pytest
from pydantic import ValidationError

from docsense.eval.schema import (
    MAX_RELEVANCE,
    CreditedHits,
    EvalReport,
    GoldExample,
    LabeledSpan,
    QueryScore,
    RetrievedRef,
)


def _span(**overrides):
    base = dict(
        arxiv_id="2301.00001",
        page_start=2,
        page_end=3,
        relevance=1,
        note=None,
        source_chunk_id="2301.00001_chunk_0001",
    )
    base.update(overrides)
    return LabeledSpan(**base)


def _gold_example(**overrides):
    base = dict(
        query_id="q001",
        query="what is the method",
        query_type="method",
        difficulty="medium",
        unanswerable=False,
        spans=[_span()],
        tags=["foo"],
        labeled_by="tester",
        labeled_at="2024-01-01T00:00:00+00:00",
        pooled_from=["dense"],
        pool_k=20,
    )
    base.update(overrides)
    return GoldExample(**base)


# --- Round-trips (§6.5) -------------------------------------------------------


def test_labeled_span_round_trips():
    span = _span()
    assert LabeledSpan.model_validate_json(span.model_dump_json()) == span


def test_gold_example_with_multiple_spans_round_trips():
    example = _gold_example(
        spans=[_span(page_start=2, page_end=3), _span(page_start=8, page_end=8, relevance=2)]
    )
    assert GoldExample.model_validate_json(example.model_dump_json()) == example


def test_unanswerable_gold_example_with_no_spans_round_trips():
    example = _gold_example(unanswerable=True, spans=[])
    assert GoldExample.model_validate_json(example.model_dump_json()) == example


def test_credited_hits_round_trips():
    credited = CreditedHits(gains=[0, 2, 0], first_covered_ranks=[1, None], ideal_gains=[2, 1])
    assert CreditedHits.model_validate_json(credited.model_dump_json()) == credited


def test_retrieved_ref_round_trips():
    ref = RetrievedRef(
        chunk_id="2301.00001_chunk_0001",
        arxiv_id="2301.00001",
        page_start=1,
        page_end=1,
        rank=0,
        score=0.9,
        credited_relevance=2,
    )
    assert RetrievedRef.model_validate_json(ref.model_dump_json()) == ref


def test_query_score_round_trips():
    score = QueryScore(
        query_id="q001",
        query="what is x",
        query_type="factoid",
        difficulty="easy",
        num_relevant_spans=1,
        unanswerable=False,
        scored=True,
        metrics={"recall@5": 1.0, "mrr": 1.0},
        first_relevant_rank=0,
        retrieved=[],
    )
    assert QueryScore.model_validate_json(score.model_dump_json()) == score


def test_eval_report_round_trips():
    report = EvalReport(
        run_id="20240101T000000Z_baseline",
        started_at="2024-01-01T00:00:00+00:00",
        finished_at="2024-01-01T00:00:01+00:00",
        gold_path="data/eval/gold_set.jsonl",
        gold_sha256="a" * 64,
        num_queries=0,
        num_scored=0,
        num_unanswerable=0,
        retriever="dense",
        retriever_config="baseline",
        variant_name="baseline",
        embedder_model="hash",
        index_fingerprint="b" * 64,
        k_values=[1, 3, 5, 10],
        primary_k=5,
        binary_relevance_threshold=1,
        retrieval_top_k=10,
        seed=0,
        library_versions={"numpy": "1.0"},
        metrics={"recall@5": None},
        metrics_by_slice={},
        mean_latency_ms=None,
        per_query=[],
    )
    assert EvalReport.model_validate_json(report.model_dump_json()) == report


# --- Validator rejections (§6.6) ----------------------------------------------


def test_page_start_greater_than_page_end_rejected():
    with pytest.raises(ValidationError):
        _span(page_start=5, page_end=4)


@pytest.mark.parametrize("relevance", [-1, MAX_RELEVANCE + 1])
def test_out_of_range_relevance_rejected(relevance):
    with pytest.raises(ValidationError):
        _span(relevance=relevance)


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_rejected(query):
    with pytest.raises(ValidationError):
        _gold_example(query=query)


@pytest.mark.parametrize("labeled_by", ["", "   "])
def test_blank_labeled_by_rejected(labeled_by):
    with pytest.raises(ValidationError):
        _gold_example(labeled_by=labeled_by)


def test_unanswerable_true_with_relevant_span_rejected():
    with pytest.raises(ValidationError):
        _gold_example(unanswerable=True, spans=[_span(relevance=1)])


def test_unanswerable_false_with_no_spans_rejected():
    with pytest.raises(ValidationError):
        _gold_example(unanswerable=False, spans=[])


def test_unanswerable_false_with_only_relevance_zero_spans_rejected():
    with pytest.raises(ValidationError):
        _gold_example(unanswerable=False, spans=[_span(relevance=0)])
