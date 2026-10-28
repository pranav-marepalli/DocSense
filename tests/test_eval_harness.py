"""Acceptance criteria §6.9-6.10: evaluate() on an empty gold set and on a
small synthetic in-test gold set, end-to-end against a fixture corpus
(hash embedder + numpy store)."""

from pathlib import Path

import pytest

from docsense.eval.harness import evaluate, score_example
from docsense.eval.schema import GoldExample, LabeledSpan
from docsense.index.build import build_index
from docsense.retrieve.dense import DenseRetriever
from tests.conftest import write_sample_chunks


def _retriever_config() -> dict:
    return {
        "variant_name": "test",
        "seed": 0,
        "record_format": "jsonl",
        "embedder": {
            "type": "hash",
            "model_name": "BAAI/bge-small-en-v1.5",
            "dim": 16,
            "normalize": True,
            "batch_size": 4,
            "device": None,
            "max_seq_length": 512,
            "query_prefix": "",
            "passage_prefix": "",
            "trust_remote_code": False,
        },
        "index": {"type": "numpy_flat_ip", "num_threads": 1},
        "retrieval": {"top_k": 10, "max_chunks_per_paper": None},
    }


def _eval_config(k_values=(1, 3, 5, 10), retrieval_top_k=10) -> dict:
    return {
        "seed": 0,
        "eval": {
            "gold_path": "unused-filled-in-by-caller",
            "runs_dir": "unused",
            "record_format": "jsonl",
            "k_values": list(k_values),
            "primary_k": 5,
            "retrieval_top_k": retrieval_top_k,
            "binary_relevance_threshold": 1,
            "per_query_detail": True,
            "slice_by": ["query_type", "difficulty"],
            "min_queries_warn": 30,
        },
    }


@pytest.fixture
def retriever(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = write_sample_chunks(chunks_path)
    index_dir = tmp_path / "index"
    config = _retriever_config()
    build_index(chunks_path, index_dir, config)
    return DenseRetriever.load(index_dir, config), chunks


def _span(chunk, relevance: int = 2):
    return LabeledSpan(
        arxiv_id=chunk.arxiv_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        relevance=relevance,
        note=None,
        source_chunk_id=chunk.chunk_id,
    )


def _example(query_id, query, spans, query_type="factoid", difficulty="easy", unanswerable=False):
    return GoldExample(
        query_id=query_id,
        query=query,
        query_type=query_type,
        difficulty=difficulty,
        unanswerable=unanswerable,
        spans=spans,
        tags=[],
        labeled_by="tester",
        labeled_at="2024-01-01T00:00:00+00:00",
        pooled_from=["dense"],
        pool_k=20,
    )


# --- §6.9: empty gold set ------------------------------------------------------


def test_evaluate_empty_gold_set_does_not_raise_and_metrics_are_none(retriever):
    dense_retriever, _ = retriever
    config = _eval_config()
    config["eval"]["gold_path"] = "does-not-matter.jsonl"

    report = evaluate(dense_retriever, [], config, retriever_config_name="offline_smoke")

    assert report.num_queries == 0
    assert report.num_scored == 0
    assert report.per_query == []
    assert all(value is None for value in report.metrics.values())
    assert set(report.metrics.keys()) == {
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "ndcg@1",
        "ndcg@3",
        "ndcg@5",
        "ndcg@10",
        "mrr",
    }


def test_evaluate_empty_gold_set_populates_provenance(retriever):
    dense_retriever, _ = retriever
    config = _eval_config()
    config["eval"]["gold_path"] = "does-not-matter.jsonl"

    report = evaluate(dense_retriever, [], config, retriever_config_name="offline_smoke")

    assert report.gold_sha256  # non-empty sha256 hex digest even for a missing/empty gold file
    assert report.index_fingerprint
    assert report.variant_name == "test"
    assert report.seed == 0
    assert report.library_versions
    assert report.embedder_model == "hash"


# --- §6.10: synthetic in-test gold set -----------------------------------------


def test_evaluate_synthetic_gold_set_end_to_end(retriever, tmp_path: Path):
    dense_retriever, chunks = retriever
    target = chunks[5]  # 2301.00002_chunk_0000's own text is its best match
    miss_target = chunks[0]

    perfect_example = _example(
        "q001", target.text, [_span(target, relevance=2)], query_type="table", difficulty="hard"
    )
    missed_example = _example(
        "q002",
        "a query about a page nothing in the corpus covers",
        [LabeledSpan(arxiv_id="9999.99999", page_start=1, page_end=1, relevance=1,
                     note=None, source_chunk_id=None)],
        query_type="factoid",
        difficulty="easy",
    )
    unanswerable_example = _example(
        "q003", "a query with no answer in the corpus", [], unanswerable=True
    )

    gold = [perfect_example, missed_example, unanswerable_example]
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text("\n".join(e.model_dump_json() for e in gold) + "\n")

    config = _eval_config()
    config["eval"]["gold_path"] = str(gold_path)

    report = evaluate(dense_retriever, gold, config, retriever_config_name="offline_smoke")

    assert report.num_queries == 3
    assert report.num_unanswerable == 1
    assert report.num_scored == 2

    metric_keys = set(report.metrics.keys())
    assert metric_keys == {
        "recall@1", "recall@3", "recall@5", "recall@10",
        "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "mrr",
    }

    perfect_score = next(q for q in report.per_query if q.query_id == "q001")
    assert perfect_score.scored
    assert perfect_score.metrics["recall@1"] == pytest.approx(1.0)
    assert perfect_score.metrics["mrr"] == pytest.approx(1.0)
    assert perfect_score.metrics["ndcg@1"] == pytest.approx(1.0)
    assert perfect_score.first_relevant_rank == 0

    missed_score = next(q for q in report.per_query if q.query_id == "q002")
    assert missed_score.scored
    for k in (1, 3, 5, 10):
        assert missed_score.metrics[f"recall@{k}"] == pytest.approx(0.0)
        assert missed_score.metrics[f"ndcg@{k}"] == pytest.approx(0.0)
    assert missed_score.metrics["mrr"] == pytest.approx(0.0)
    assert missed_score.first_relevant_rank is None

    unanswerable_score = next(q for q in report.per_query if q.query_id == "q003")
    assert not unanswerable_score.scored
    assert unanswerable_score.metrics == {}

    assert "query_type=table" in report.metrics_by_slice
    assert "query_type=factoid" in report.metrics_by_slice
    assert "difficulty=hard" in report.metrics_by_slice
    assert "difficulty=easy" in report.metrics_by_slice

    del miss_target  # unused, kept only to document which chunk is "missed" conceptually


def test_evaluate_raises_when_retrieval_top_k_below_max_k_values(retriever):
    dense_retriever, _ = retriever
    config = _eval_config(k_values=(1, 3, 5, 10), retrieval_top_k=5)
    config["eval"]["gold_path"] = "unused.jsonl"

    with pytest.raises(ValueError):
        evaluate(dense_retriever, [], config, retriever_config_name="offline_smoke")


# --- score_example unit coverage ------------------------------------------------


def test_score_example_unanswerable_returns_unscored(retriever):
    dense_retriever, _ = retriever
    example = _example("q010", "irrelevant text query", [], unanswerable=True)
    result = dense_retriever.retrieve(example.query, top_k=5)

    score = score_example(example, result, k_values=[1, 5], threshold=1, include_detail=True)

    assert score.scored is False
    assert score.metrics == {}
    assert score.unanswerable is True
    assert score.retrieved == []


# --- R4 regression: no span reaches eval.binary_relevance_threshold -------------


def test_score_example_below_threshold_is_unscored_not_a_crash(retriever):
    """An answerable example whose only labeled span is below
    `eval.binary_relevance_threshold` must not crash `score_example` -- it has
    no relevant span *at this threshold*, so it is unscorable, same as the
    unanswerable path, rather than reaching `ndcg_at_k` with an empty
    `ideal_gains` and raising deep inside metrics.py."""
    dense_retriever, chunks = retriever
    example = _example(
        "q011", chunks[0].text, [_span(chunks[0], relevance=1)], unanswerable=False
    )
    result = dense_retriever.retrieve(example.query, top_k=5)

    score = score_example(example, result, k_values=[1, 5], threshold=2, include_detail=True)

    assert score.scored is False
    assert score.metrics == {}
    assert score.unanswerable is False
    assert score.retrieved == []


def test_evaluate_excludes_below_threshold_example_from_scored_without_raising(
    retriever, tmp_path: Path
):
    dense_retriever, chunks = retriever
    example = _example(
        "q012", chunks[0].text, [_span(chunks[0], relevance=1)], unanswerable=False
    )
    config = _eval_config()
    config["eval"]["gold_path"] = str(tmp_path / "gold.jsonl")
    config["eval"]["binary_relevance_threshold"] = 2

    report = evaluate(dense_retriever, [example], config, retriever_config_name="offline_smoke")

    assert report.num_queries == 1
    assert report.num_scored == 0
    assert report.num_unanswerable == 0
    assert all(value is None for value in report.metrics.values())
