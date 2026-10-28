"""Acceptance criteria §6.7, §6.8, and the write_eval_report half of §6.11."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from docsense.eval.io import export_eval_texts, read_gold_set, write_eval_report
from docsense.eval.schema import EvalReport, GoldExample, LabeledSpan
from docsense.index.schema import IndexedChunk
from docsense.ingest.io import get_record_store

REPO_ROOT = Path(__file__).resolve().parents[1]


def _gold_line(query_id: str, query: str, labeled_by: str = "tester") -> dict:
    return GoldExample(
        query_id=query_id,
        query=query,
        query_type="factoid",
        difficulty="easy",
        unanswerable=False,
        spans=[
            LabeledSpan(
                arxiv_id="2301.00001",
                page_start=1,
                page_end=1,
                relevance=1,
                note=None,
                source_chunk_id="2301.00001_chunk_0000",
            )
        ],
        tags=[],
        labeled_by=labeled_by,
        labeled_at="2024-01-01T00:00:00+00:00",
        pooled_from=["dense"],
        pool_k=20,
    ).model_dump(mode="json")


@pytest.fixture
def store():
    return get_record_store("jsonl")


# --- read_gold_set (§6.7) -----------------------------------------------------


def test_read_gold_set_round_trips_hand_written_lines_in_order(tmp_path: Path, store):
    path = tmp_path / "gold.jsonl"
    lines = [_gold_line("q001", "first query"), _gold_line("q002", "second query")]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    examples = read_gold_set(path, store)

    assert [e.query_id for e in examples] == ["q001", "q002"]
    assert [e.query for e in examples] == ["first query", "second query"]


def test_read_gold_set_missing_file_returns_empty(tmp_path: Path, store):
    assert read_gold_set(tmp_path / "does_not_exist.jsonl", store) == []


def test_read_gold_set_zero_byte_file_returns_empty(tmp_path: Path, store):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert read_gold_set(path, store) == []


def test_read_gold_set_duplicate_query_id_raises(tmp_path: Path, store):
    path = tmp_path / "gold.jsonl"
    lines = [_gold_line("q001", "first query"), _gold_line("q001", "a different query")]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    with pytest.raises(ValueError):
        read_gold_set(path, store)


def test_read_gold_set_duplicate_query_text_case_and_whitespace_insensitive_raises(
    tmp_path: Path, store
):
    path = tmp_path / "gold.jsonl"
    lines = [
        _gold_line("q001", "What is   the Method"),
        _gold_line("q002", "what is the method"),
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    with pytest.raises(ValueError):
        read_gold_set(path, store)


def test_read_gold_set_malformed_line_raises_validation_error(tmp_path: Path, store):
    path = tmp_path / "gold.jsonl"
    record = _gold_line("q001", "first query")
    del record["labeled_by"]
    path.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValidationError):
        read_gold_set(path, store)


# --- The committed gold set is empty at phase end (§6.8) ----------------------


def test_committed_gold_set_is_zero_bytes_and_reads_as_empty(store):
    real_gold_path = REPO_ROOT / "data" / "eval" / "gold_set.jsonl"
    assert real_gold_path.exists()
    assert real_gold_path.stat().st_size == 0
    assert read_gold_set(real_gold_path, store) == []


# --- write_eval_report leakage guard (§6.11) ----------------------------------


def _report(gold_path: str, run_id: str = "20240101T000000Z_baseline") -> EvalReport:
    return EvalReport(
        run_id=run_id,
        started_at="2024-01-01T00:00:00+00:00",
        finished_at="2024-01-01T00:00:01+00:00",
        gold_path=gold_path,
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
        library_versions={},
        metrics={},
        metrics_by_slice={},
        mean_latency_ms=None,
        per_query=[],
    )


def test_write_eval_report_writes_run_id_json(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    report = _report(gold_path=str(tmp_path / "gold_set.jsonl"))

    path = write_eval_report(report, runs_dir)

    assert path == runs_dir / f"{report.run_id}.json"
    assert EvalReport.model_validate_json(path.read_text()) == report


def test_write_eval_report_refuses_configured_gold_path(tmp_path: Path):
    # gold_path deliberately has no "gold" in its name, so this exercises the
    # exact-path-match branch rather than the *gold* filename branch below.
    gold_path = tmp_path / "labels.json"
    report = _report(gold_path=str(gold_path), run_id="labels")
    # runs_dir chosen so runs_dir/<run_id>.json literally resolves to gold_path
    # (gold_path's suffix is irrelevant to the path-equality check).
    runs_dir = tmp_path

    with pytest.raises(ValueError):
        write_eval_report(report, runs_dir)


def test_write_eval_report_refuses_any_gold_named_file(tmp_path: Path):
    report = _report(gold_path=str(tmp_path / "unrelated_gold.jsonl"), run_id="my_gold_run")
    runs_dir = tmp_path / "runs"

    with pytest.raises(ValueError):
        write_eval_report(report, runs_dir)


# --- export_eval_texts ---------------------------------------------------------


def test_export_eval_texts_writes_query_and_passage_records(tmp_path: Path, store):
    example = GoldExample.model_validate(_gold_line("q001", "what is the method"))
    chunk = IndexedChunk(
        row_id=0,
        chunk_id="2301.00001_chunk_0000",
        arxiv_id="2301.00001",
        chunk_index=0,
        page_start=1,
        page_end=1,
        text="the method section describes the approach",
        token_count=7,
        source="text",
    )
    unrelated_chunk = IndexedChunk(
        row_id=1,
        chunk_id="2301.00002_chunk_0000",
        arxiv_id="2301.00002",
        chunk_index=0,
        page_start=1,
        page_end=1,
        text="a completely unrelated passage",
        token_count=4,
        source="text",
    )
    out_path = tmp_path / "eval_texts.jsonl"

    count = export_eval_texts([example], [chunk, unrelated_chunk], out_path, store)

    records = store.read_all(out_path)
    assert count == len(records) == 2
    kinds = {r["kind"] for r in records}
    assert kinds == {"query", "passage"}
    query_records = [r for r in records if r["kind"] == "query"]
    passage_records = [r for r in records if r["kind"] == "passage"]
    assert query_records[0]["text"] == "what is the method"
    assert passage_records[0]["chunk_id"] == "2301.00001_chunk_0000"
    assert all("text" in r for r in records)


# --- R1: export_eval_texts refuses a gold-shaped out_path ---------------------


def test_export_eval_texts_refuses_configured_gold_path(tmp_path: Path, store):
    """Reproduces the reviewer's empirical repro: exporting straight over the
    configured gold path must raise, not silently unlink() and overwrite the
    human gold set with derived (query_id, text) records."""
    gold_path = tmp_path / "gold_set.jsonl"
    gold_path.write_text('{"not": "touched"}\n')
    original_bytes = gold_path.read_bytes()

    with pytest.raises(ValueError):
        export_eval_texts([], [], gold_path, store, gold_path=gold_path)

    assert gold_path.read_bytes() == original_bytes


def test_export_eval_texts_refuses_any_gold_named_out_path_even_without_gold_path_arg(
    tmp_path: Path, store
):
    """The `*gold*` filename check fires even when the caller does not pass
    `gold_path` (e.g. an older call site), matching `write_eval_report`'s
    always-on name heuristic."""
    out_path = tmp_path / "unrelated_gold_copy.jsonl"

    with pytest.raises(ValueError):
        export_eval_texts([], [], out_path, store)

    assert not out_path.exists()
