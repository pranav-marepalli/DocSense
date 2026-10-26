"""Acceptance criteria 11, 12, 13: ingest_pdf, ingest_corpus, and the ingestion report."""

import shutil
from pathlib import Path

import pymupdf as fitz

from docsense.ingest.io import get_record_store, read_chunks
from docsense.ingest.pipeline import ingest_corpus, ingest_pdf
from docsense.ingest.schema import IngestReport


def _test_config(**overrides) -> dict:
    config = {
        "record_format": "jsonl",
        "chunk_size_tokens": 50,
        "chunk_overlap_tokens": 10,
        "min_chunk_tokens": 5,
        "min_text_chars_per_page": 10,
        "table_detection": True,
        "figure_extraction": True,
        "figure_dpi": 150,
        "caption_search_radius_pts": 40,
        "skip_on_error": True,
    }
    config.update(overrides)
    return config


def _make_valid_pdf(path: Path, body: str) -> None:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), body)
    doc.save(path)
    doc.close()


def test_ingest_pdf_builds_paper_document(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    _make_valid_pdf(pdf_path, "Some real paper content. " * 40)

    config = _test_config()
    doc = ingest_pdf(pdf_path, "2301.00001", tmp_path / "images", config)

    assert doc.arxiv_id == "2301.00001"
    assert doc.num_pages == 1
    assert len(doc.pages) == 1
    assert len(doc.chunks) >= 1
    assert all(c.arxiv_id == "2301.00001" for c in doc.chunks)


def test_ingest_corpus_skips_truncated_pdf_and_continues(tmp_path: Path):
    pdf_dir = tmp_path / "raw"
    pdf_dir.mkdir()
    _make_valid_pdf(pdf_dir / "2301.00001.pdf", "First valid paper. " * 40)
    _make_valid_pdf(pdf_dir / "2301.00002.pdf", "Second valid paper. " * 40)
    (pdf_dir / "2301.00003.pdf").write_bytes(b"%PDF-1.4 not really a pdf")

    out_dir = tmp_path / "parsed"
    config = _test_config(skip_on_error=True)

    report = ingest_corpus(out_dir, config, pdf_dir=pdf_dir)

    assert report.papers_total == 3
    assert report.papers_ok == 2
    assert report.papers_failed == 1

    ok_ids = {s.arxiv_id for s in report.papers if s.status == "ok"}
    assert ok_ids == {"2301.00001", "2301.00002"}
    assert (out_dir / "papers" / "2301.00001.json").exists()
    assert (out_dir / "papers" / "2301.00002.json").exists()
    assert not (out_dir / "papers" / "2301.00003.json").exists()

    failed_status = next(s for s in report.papers if s.status == "error")
    assert failed_status.arxiv_id == "2301.00003"
    assert failed_status.error  # non-empty string naming the cause
    assert len(failed_status.error) > 0


def test_ingest_corpus_report_aggregates_match_paper_statuses(
    tmp_path: Path, pdf_with_table_and_image: Path, three_page_pdf: Path
):
    """R1: uses a fixture corpus that actually contains a table, a figure, and a
    low-text-layer page, so tables_total/figures_total/low_text_layer_pages and
    tables_by_method are exercised rather than vacuously 0 == 0."""
    pdf_dir = tmp_path / "raw"
    pdf_dir.mkdir()
    shutil.copy(pdf_with_table_and_image, pdf_dir / "2301.00001.pdf")  # table + figure
    shutil.copy(three_page_pdf, pdf_dir / "2301.00002.pdf")  # near-empty page 2
    (pdf_dir / "2301.00003.pdf").write_bytes(b"%PDF-1.4 not really a pdf")

    out_dir = tmp_path / "parsed"
    config = _test_config(skip_on_error=True)
    report = ingest_corpus(out_dir, config, pdf_dir=pdf_dir)

    assert report.papers_ok == 2
    assert report.papers_failed == 1
    assert report.tables_total > 0
    assert report.figures_total > 0
    assert report.low_text_layer_pages > 0

    # ingest_report.json validates as IngestReport
    persisted = IngestReport.model_validate_json((out_dir / "ingest_report.json").read_text())
    assert persisted == report

    assert report.papers_ok == sum(1 for s in report.papers if s.status == "ok")
    assert report.papers_failed == sum(1 for s in report.papers if s.status == "error")
    assert report.total_pages == sum(s.num_pages for s in report.papers)
    assert report.total_chunks == sum(s.num_chunks for s in report.papers)
    assert report.tables_total == sum(report.tables_by_method.values())
    # tables_by_method is a separate counter from the per-paper num_tables fields
    # (R1): assert the two cannot have diverged.
    assert report.tables_total == sum(s.num_tables for s in report.papers)
    assert report.figures_total == sum(s.num_figures for s in report.papers)

    from docsense.ingest.io import read_paper

    def _low_text_pages(arxiv_id: str) -> int:
        paper = read_paper(out_dir / "papers" / f"{arxiv_id}.json")
        return sum(1 for page in paper.pages if page.low_text_layer)

    low_text_from_papers = sum(
        _low_text_pages(s.arxiv_id) for s in report.papers if s.status == "ok"
    )
    assert report.low_text_layer_pages == low_text_from_papers


def test_write_chunks_output_matches_chunk_counts(tmp_path: Path):
    pdf_dir = tmp_path / "raw"
    pdf_dir.mkdir()
    _make_valid_pdf(pdf_dir / "2301.00001.pdf", "First valid paper. " * 40)
    _make_valid_pdf(pdf_dir / "2301.00002.pdf", "Second valid paper. " * 40)

    out_dir = tmp_path / "parsed"
    config = _test_config()
    report = ingest_corpus(out_dir, config, pdf_dir=pdf_dir)

    chunks_path = out_dir / "chunks.jsonl"
    lines = chunks_path.read_text().strip().split("\n")
    assert len(lines) == sum(s.num_chunks for s in report.papers if s.status == "ok")

    store = get_record_store("jsonl")
    chunks = read_chunks(chunks_path, store)
    assert len(chunks) == len(lines)


# --- R3: ingest_corpus must rebuild chunks.jsonl, not append to it ----------


def test_ingest_corpus_rerun_does_not_duplicate_chunks(tmp_path: Path):
    """ingest_corpus is a whole-corpus rebuild (unlike the append-only fetch step): an
    identical second run over the same out_dir must not duplicate chunk lines."""
    pdf_dir = tmp_path / "raw"
    pdf_dir.mkdir()
    _make_valid_pdf(pdf_dir / "2301.00001.pdf", "First valid paper. " * 40)

    out_dir = tmp_path / "parsed"
    config = _test_config()

    first_report = ingest_corpus(out_dir, config, pdf_dir=pdf_dir)
    chunks_path = out_dir / "chunks.jsonl"
    first_lines = chunks_path.read_text().strip().split("\n")
    assert len(first_lines) == first_report.total_chunks

    second_report = ingest_corpus(out_dir, config, pdf_dir=pdf_dir)
    second_lines = chunks_path.read_text().strip().split("\n")

    assert len(second_lines) == len(first_lines) == second_report.total_chunks


# --- R5: manifest-driven ingestion (the canonical path) ---------------------


def test_ingest_corpus_manifest_driven_resolves_pdf_path_against_repo_root(
    tmp_path: Path, monkeypatch
):
    """`ingest_corpus(manifest_path=...)` is the canonical path (§3): manifest
    `pdf_path`s are relative to the repo root and must resolve correctly even when
    the manifest/PDFs don't live under the real repo root (simulated here by
    monkeypatching `pipeline.REPO_ROOT` to a fake root under `tmp_path`)."""
    from docsense.ingest import pipeline as pipeline_module
    from docsense.ingest.io import append_manifest, get_record_store
    from docsense.ingest.schema import ManifestRecord

    fake_repo_root = tmp_path / "fake_repo"
    monkeypatch.setattr(pipeline_module, "REPO_ROOT", fake_repo_root)

    pdf_dir = fake_repo_root / "data" / "raw"
    pdf_dir.mkdir(parents=True)
    _make_valid_pdf(pdf_dir / "2301.00001.pdf", "First valid paper. " * 40)
    _make_valid_pdf(pdf_dir / "2301.00002.pdf", "Second valid paper. " * 40)

    manifest_path = fake_repo_root / "data" / "manifest.jsonl"
    store = get_record_store("jsonl")
    for arxiv_id in ("2301.00001", "2301.00002"):
        append_manifest(
            ManifestRecord(
                arxiv_id=arxiv_id,
                version=1,
                title="T",
                categories=["cs.CL"],
                query_category="cs.CL",
                published="2023-01-10T09:00:00Z",
                pdf_path=f"data/raw/{arxiv_id}.pdf",
                sha256="a" * 64,
                num_bytes=1,
                fetched_at="2024-01-01T00:00:00+00:00",
            ),
            manifest_path,
            store,
        )

    out_dir = tmp_path / "parsed"
    config = _test_config()
    report = ingest_corpus(out_dir, config, manifest_path=manifest_path)

    assert report.papers_total == 2
    assert report.papers_ok == 2
    ok_ids = {s.arxiv_id for s in report.papers if s.status == "ok"}
    assert ok_ids == {"2301.00001", "2301.00002"}
    assert (out_dir / "papers" / "2301.00001.json").exists()
    assert (out_dir / "papers" / "2301.00002.json").exists()

    # source_pdf is stored relative to the (fake) repo root too, per schema contract
    from docsense.ingest.io import read_paper

    paper = read_paper(out_dir / "papers" / "2301.00001.json")
    assert not Path(paper.source_pdf).is_absolute()
    assert paper.source_pdf == "data/raw/2301.00001.pdf"
