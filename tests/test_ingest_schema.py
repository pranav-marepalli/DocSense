"""Acceptance criterion 9: every schema.py model round-trips
model_dump_json() -> model_validate_json() to an equal object."""

from docsense.ingest.schema import (
    Block,
    Chunk,
    FetchReport,
    Figure,
    IngestReport,
    ManifestRecord,
    Page,
    PaperDocument,
    PaperStatus,
    Table,
)


def _roundtrip(model):
    cls = type(model)
    return cls.model_validate_json(model.model_dump_json())


def test_block_roundtrips():
    model = Block(bbox=(1.0, 2.0, 3.0, 4.0), text="hello")
    assert _roundtrip(model) == model


def test_page_roundtrips():
    model = Page(
        page_number=1,
        text="hello world",
        blocks=[Block(bbox=(0.0, 0.0, 10.0, 10.0), text="hello world")],
        char_count=11,
        low_text_layer=False,
    )
    assert _roundtrip(model) == model


def test_chunk_roundtrips():
    model = Chunk(
        chunk_id="2301.00001_chunk_0000",
        arxiv_id="2301.00001",
        chunk_index=0,
        page_start=1,
        page_end=2,
        text="some chunk text",
        token_count=3,
        source="text",
    )
    assert _roundtrip(model) == model


def test_table_roundtrips_with_and_without_markdown():
    with_markdown = Table(
        table_id="2301.00001_table_000",
        page_number=1,
        bbox=(0.0, 0.0, 100.0, 50.0),
        caption="Table 1: a caption",
        markdown="|a|b|\n|---|---|\n|1|2|",
        raw_text="a b\n1 2",
        method="find_tables",
    )
    assert _roundtrip(with_markdown) == with_markdown

    fallback = Table(
        table_id="2301.00001_table_001",
        page_number=2,
        bbox=(0.0, 0.0, 100.0, 50.0),
        caption=None,
        markdown=None,
        raw_text="a b\n1 2",
        method="raw_text_fallback",
    )
    assert _roundtrip(fallback) == fallback


def test_figure_roundtrips():
    model = Figure(
        figure_id="2301.00001_figure_000",
        page_number=1,
        bbox=(0.0, 0.0, 100.0, 100.0),
        caption="Figure 1: a caption",
        image_path="images/2301.00001/figure_000.png",
    )
    assert _roundtrip(model) == model


def test_paper_document_roundtrips():
    model = PaperDocument(
        arxiv_id="2301.00001",
        source_pdf="data/raw/2301.00001v1.pdf",
        title="A Great Paper",
        num_pages=2,
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
        chunks=[
            Chunk(
                chunk_id="2301.00001_chunk_0000",
                arxiv_id="2301.00001",
                chunk_index=0,
                page_start=1,
                page_end=1,
                text="hello",
                token_count=1,
                source="text",
            )
        ],
    )
    assert _roundtrip(model) == model


def test_manifest_record_roundtrips():
    model = ManifestRecord(
        arxiv_id="2301.00001",
        version=1,
        title="A Great Paper",
        categories=["cs.CL", "cs.LG"],
        query_category="cs.CL",
        published="2023-01-10T09:00:00Z",
        pdf_path="data/raw/2301.00001v1.pdf",
        sha256="a" * 64,
        num_bytes=12345,
        fetched_at="2024-01-01T00:00:00+00:00",
    )
    assert _roundtrip(model) == model


def test_paper_status_roundtrips():
    model = PaperStatus(
        arxiv_id="2301.00001",
        status="error",
        error="ValueError: bad pdf",
        num_pages=0,
        num_chunks=0,
        num_tables=0,
        num_figures=0,
        num_low_text_pages=0,
    )
    assert _roundtrip(model) == model


def test_ingest_report_roundtrips():
    model = IngestReport(
        started_at="2024-01-01T00:00:00+00:00",
        finished_at="2024-01-01T00:05:00+00:00",
        extractor_version="pymupdf-1.24.10",
        papers_total=1,
        papers_ok=1,
        papers_failed=0,
        total_pages=2,
        total_chunks=1,
        tables_total=0,
        tables_by_method={"find_tables": 0, "raw_text_fallback": 0},
        figures_total=0,
        low_text_layer_pages=0,
        papers=[
            PaperStatus(
                arxiv_id="2301.00001",
                status="ok",
                error=None,
                num_pages=2,
                num_chunks=1,
                num_tables=0,
                num_figures=0,
                num_low_text_pages=0,
            )
        ],
    )
    assert _roundtrip(model) == model


def test_fetch_report_roundtrips():
    model = FetchReport(
        started_at="2024-01-01T00:00:00+00:00",
        finished_at="2024-01-01T00:05:00+00:00",
        search_queries={"cs.CL": "cat:cs.CL AND submittedDate:[202301010000 TO 202312312359]"},
        max_papers=8,
        category_targets={"cs.CL": 2, "cs.LG": 2, "cs.CV": 2, "cs.IR": 2},
        already_in_manifest=0,
        newly_fetched=8,
        newly_fetched_by_category={"cs.CL": 2, "cs.LG": 2, "cs.CV": 2, "cs.IR": 2},
        failed=[{"arxiv_id": "2301.99999", "error": "URLError: boom"}],
        api_requests=4,
    )
    assert _roundtrip(model) == model
