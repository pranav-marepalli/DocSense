"""Acceptance criterion 10 (RecordStore) plus write_*/read_* round-trip coverage."""

from pathlib import Path

import pytest

from docsense.ingest.io import (
    JsonlRecordStore,
    append_manifest,
    get_record_store,
    read_chunks,
    read_manifest,
    read_paper,
    write_chunks,
    write_ingest_report,
    write_paper,
)
from docsense.ingest.schema import (
    Block,
    Chunk,
    IngestReport,
    ManifestRecord,
    Page,
    PaperDocument,
    PaperStatus,
)


def test_jsonl_record_store_roundtrip_and_count(tmp_path: Path):
    store = JsonlRecordStore()
    path = tmp_path / "nested" / "records.jsonl"

    assert store.read_all(path) == []
    assert store.count(path) == 0

    records = [{"a": 1}, {"a": 2}]
    store.append(records, path)
    assert store.read_all(path) == records
    assert store.count(path) == len(records)

    # two appends accumulate rather than truncate
    more_records = [{"a": 3}]
    store.append(more_records, path)
    assert store.read_all(path) == records + more_records
    assert store.count(path) == 3


def test_get_record_store_known_and_unknown():
    store = get_record_store("jsonl")
    assert isinstance(store, JsonlRecordStore)
    assert store.extension == "jsonl"

    with pytest.raises(ValueError):
        get_record_store("nope")


def test_write_and_read_chunks(tmp_path: Path):
    store = get_record_store("jsonl")
    path = tmp_path / "chunks.jsonl"
    chunks = [
        Chunk(
            chunk_id=f"2301.00001_chunk_{i:04d}",
            arxiv_id="2301.00001",
            chunk_index=i,
            page_start=1,
            page_end=1,
            text=f"chunk {i}",
            token_count=2,
            source="text",
        )
        for i in range(3)
    ]
    write_chunks(chunks, path, store)
    assert read_chunks(path, store) == chunks

    # one JSON object per line
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 3


def test_append_and_read_manifest(tmp_path: Path):
    store = get_record_store("jsonl")
    path = tmp_path / "manifest.jsonl"

    assert read_manifest(path, store) == []

    record = ManifestRecord(
        arxiv_id="2301.00001",
        version=1,
        title="A Great Paper",
        categories=["cs.CL"],
        query_category="cs.CL",
        published="2023-01-10T09:00:00Z",
        pdf_path="data/raw/2301.00001v1.pdf",
        sha256="a" * 64,
        num_bytes=100,
        fetched_at="2024-01-01T00:00:00+00:00",
    )
    append_manifest(record, path, store)
    assert read_manifest(path, store) == [record]


def test_write_and_read_paper(tmp_path: Path):
    doc = PaperDocument(
        arxiv_id="2301.00001",
        source_pdf="data/raw/2301.00001v1.pdf",
        title="A Great Paper",
        num_pages=1,
        extracted_at="2024-01-01T00:00:00+00:00",
        extractor_version="pymupdf-1.24.10",
        pages=[
            Page(
                page_number=1,
                text="hello",
                blocks=[Block(bbox=(0.0, 0.0, 1.0, 1.0), text="hello")],
                char_count=5,
                low_text_layer=False,
            )
        ],
        tables=[],
        figures=[],
        chunks=[],
    )
    path = write_paper(doc, tmp_path)
    assert path == tmp_path / "papers" / "2301.00001.json"
    assert read_paper(path) == doc


def test_write_ingest_report(tmp_path: Path):
    report = IngestReport(
        started_at="2024-01-01T00:00:00+00:00",
        finished_at="2024-01-01T00:05:00+00:00",
        extractor_version="pymupdf-1.24.10",
        papers_total=1,
        papers_ok=1,
        papers_failed=0,
        total_pages=1,
        total_chunks=0,
        tables_total=0,
        tables_by_method={"find_tables": 0, "raw_text_fallback": 0},
        figures_total=0,
        low_text_layer_pages=0,
        papers=[
            PaperStatus(
                arxiv_id="2301.00001",
                status="ok",
                error=None,
                num_pages=1,
                num_chunks=0,
                num_tables=0,
                num_figures=0,
                num_low_text_pages=0,
            )
        ],
    )
    path = write_ingest_report(report, tmp_path)
    assert path == tmp_path / "ingest_report.json"
    assert IngestReport.model_validate_json(path.read_text()) == report
