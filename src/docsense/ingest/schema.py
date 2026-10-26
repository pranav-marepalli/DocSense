"""Pydantic models for the ingestion pipeline. No I/O here.

Page and Chunk are the two models the whole pipeline is built around: Page is one
PDF page with its text and layout blocks, and Chunk is the retrieval-sized,
page-cited passage that Phase 2 indexes. The rest (Table, Figure, PaperDocument,
ManifestRecord, PaperStatus, IngestReport, FetchReport) are flat records that
round-trip through `model_dump_json()` / `model_validate_json()`.
"""

from typing import Literal

from pydantic import BaseModel

BBox = tuple[float, float, float, float]


class Block(BaseModel):
    """One text block from `page.get_text("blocks")`: a bounding box (PDF points) and its text."""

    bbox: BBox
    text: str


class Page(BaseModel):
    """One page. page_number is 1-indexed and is the citation unit everywhere downstream."""

    page_number: int
    text: str
    blocks: list[Block]
    char_count: int
    low_text_layer: bool


class Chunk(BaseModel):
    """A retrieval-sized passage. The record Phase 2 indexes."""

    chunk_id: str
    arxiv_id: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    token_count: int
    source: Literal["text"]


class Table(BaseModel):
    """A detected (or fallback-extracted) table on one page."""

    table_id: str
    page_number: int
    bbox: BBox
    caption: str | None
    markdown: str | None
    raw_text: str
    method: Literal["find_tables", "raw_text_fallback"]


class Figure(BaseModel):
    """A cropped figure image plus its nearby caption, if any."""

    figure_id: str
    page_number: int
    bbox: BBox
    caption: str | None
    image_path: str


class PaperDocument(BaseModel):
    """Full per-paper extraction record, written to data/parsed/papers/<arxiv_id>.json."""

    arxiv_id: str
    source_pdf: str
    title: str | None
    num_pages: int
    extracted_at: str
    extractor_version: str
    pages: list[Page]
    tables: list[Table]
    figures: list[Figure]
    chunks: list[Chunk]


class ManifestRecord(BaseModel):
    """One fetched paper, committed to data/manifest.jsonl. The corpus is rebuildable from these."""

    arxiv_id: str
    version: int
    title: str
    categories: list[str]
    query_category: str
    published: str
    pdf_path: str
    sha256: str
    num_bytes: int
    fetched_at: str


class PaperStatus(BaseModel):
    """One row of the ingestion report, one per attempted paper."""

    arxiv_id: str
    status: Literal["ok", "error"]
    error: str | None
    num_pages: int
    num_chunks: int
    num_tables: int
    num_figures: int
    num_low_text_pages: int


class IngestReport(BaseModel):
    """One ingestion run, written to data/parsed/ingest_report.json."""

    started_at: str
    finished_at: str
    extractor_version: str
    papers_total: int
    papers_ok: int
    papers_failed: int
    total_pages: int
    total_chunks: int
    tables_total: int
    tables_by_method: dict[str, int]
    figures_total: int
    low_text_layer_pages: int
    papers: list[PaperStatus]


class FetchReport(BaseModel):
    """One fetch run. Printed by the CLI; not persisted to disk."""

    started_at: str
    finished_at: str
    search_queries: dict[str, str]
    max_papers: int
    category_targets: dict[str, int]
    already_in_manifest: int
    newly_fetched: int
    newly_fetched_by_category: dict[str, int]
    failed: list[dict[str, str]]
    api_requests: int
