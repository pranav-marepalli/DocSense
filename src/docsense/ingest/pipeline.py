"""Per-paper and corpus-wide extraction driver.

`ingest_pdf` turns one PDF into an in-memory `PaperDocument`; `ingest_corpus`
runs that over a manifest or a directory of PDFs, writing the per-paper JSON,
the corpus-wide chunks file, and a final `IngestReport`.
"""

import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pymupdf as fitz

from docsense.ingest.chunk import chunk_pages
from docsense.ingest.io import (
    RecordStore,
    get_record_store,
    read_manifest,
    write_chunks,
    write_ingest_report,
    write_paper,
)
from docsense.ingest.pdf_extract import (
    extract_figures,
    extract_pages,
    extract_tables,
    extract_title,
)
from docsense.ingest.schema import IngestReport, PaperDocument, PaperStatus

logger = logging.getLogger(__name__)

ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
EXTRACTOR_VERSION = f"pymupdf-{fitz.__version__}"

# src/docsense/ingest/pipeline.py -> repo root (same pattern as docsense.config.CONFIG_DIR).
REPO_ROOT = Path(__file__).resolve().parents[3]


def _relative_to_repo_root(path: Path) -> str:
    """Best-effort path relative to `REPO_ROOT` (may contain `..` if `path` is outside it)."""
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return str(path)


def _resolve_against_repo_root(path: Path) -> Path:
    """Resolve a manifest `pdf_path` (relative to the repo root per schema contract) to an
    absolute path usable for opening the file."""
    return path if path.is_absolute() else (REPO_ROOT / path)


def ingest_pdf(pdf_path: Path, arxiv_id: str, image_dir: Path, config: dict) -> PaperDocument:
    """Open `pdf_path` with fitz, run extraction and chunking, return a `PaperDocument`.

    Does not write to disk.
    """
    with fitz.open(pdf_path) as doc:
        title = extract_title(doc)
        pages = extract_pages(doc, config["min_text_chars_per_page"])
        tables = (
            extract_tables(doc, arxiv_id, config["caption_search_radius_pts"])
            if config["table_detection"]
            else []
        )
        figures = (
            extract_figures(
                doc,
                arxiv_id,
                image_dir,
                config["figure_dpi"],
                config["caption_search_radius_pts"],
            )
            if config["figure_extraction"]
            else []
        )
        num_pages = doc.page_count

    chunks = chunk_pages(
        pages,
        arxiv_id,
        config["chunk_size_tokens"],
        config["chunk_overlap_tokens"],
        config["min_chunk_tokens"],
    )

    return PaperDocument(
        arxiv_id=arxiv_id,
        source_pdf=_relative_to_repo_root(pdf_path),
        title=title,
        num_pages=num_pages,
        extracted_at=datetime.now(UTC).isoformat(),
        extractor_version=EXTRACTOR_VERSION,
        pages=pages,
        tables=tables,
        figures=figures,
        chunks=chunks,
    )


def _arxiv_id_from_stem(stem: str) -> str:
    """Return `stem` if it looks like an arXiv id, else `stem` with a warning logged."""
    if not ARXIV_ID_RE.match(stem):
        logger.warning("filename stem %r does not look like an arXiv id; using it as-is", stem)
    return stem


def _resolve_inputs(
    manifest_path: Path | None, pdf_dir: Path | None, store: RecordStore
) -> list[tuple[str, Path]]:
    """Return (arxiv_id, pdf_path) pairs: from the manifest when given, else glob `pdf_dir/*.pdf`.

    Manifest-driven `pdf_path`s are relative to the repo root (the schema contract)
    and are resolved against `REPO_ROOT` here so the file can actually be opened.
    """
    if manifest_path is not None:
        records = read_manifest(manifest_path, store)
        return [
            (record.arxiv_id, _resolve_against_repo_root(Path(record.pdf_path)))
            for record in records
        ]

    if pdf_dir is not None:
        return [(_arxiv_id_from_stem(path.stem), path) for path in sorted(pdf_dir.glob("*.pdf"))]

    raise ValueError("Either manifest_path or pdf_dir must be given")


def ingest_corpus(
    out_dir: Path,
    config: dict,
    manifest_path: Path | None = None,
    pdf_dir: Path | None = None,
) -> IngestReport:
    """Extract every paper named by the manifest (canonical) or glob `pdf_dir/*.pdf`.

    Per paper: `ingest_pdf`, `write_paper`, append chunks to
    `out_dir/chunks.<store.extension>`; on exception log it and record a
    `PaperStatus` with `status="error"`, continuing when
    `config["skip_on_error"]` else re-raising. Finally writes and returns the
    `IngestReport` whose aggregates are summed from its `PaperStatus` rows.

    Ingestion is a whole-corpus rebuild, unlike the append-only fetch step: any
    existing `chunks.<ext>` at `out_dir` is truncated at the start of the run so a
    re-run over the same `out_dir` does not duplicate chunks.
    """
    started_at = datetime.now(UTC).isoformat()
    store = get_record_store(config["record_format"])
    image_dir = out_dir / "images"
    chunks_path = out_dir / f"chunks.{store.extension}"
    if chunks_path.exists():
        chunks_path.unlink()

    inputs = _resolve_inputs(manifest_path, pdf_dir, store)
    statuses: list[PaperStatus] = []
    tables_by_method: dict[str, int] = {"find_tables": 0, "raw_text_fallback": 0}

    for arxiv_id, pdf_path in inputs:
        try:
            paper = ingest_pdf(pdf_path, arxiv_id, image_dir, config)
            write_paper(paper, out_dir)
            write_chunks(paper.chunks, chunks_path, store)

            for table in paper.tables:
                tables_by_method[table.method] += 1

            statuses.append(
                PaperStatus(
                    arxiv_id=arxiv_id,
                    status="ok",
                    error=None,
                    num_pages=paper.num_pages,
                    num_chunks=len(paper.chunks),
                    num_tables=len(paper.tables),
                    num_figures=len(paper.figures),
                    num_low_text_pages=sum(1 for page in paper.pages if page.low_text_layer),
                )
            )
        except Exception as exc:
            logger.exception("failed to ingest %s", arxiv_id)
            statuses.append(
                PaperStatus(
                    arxiv_id=arxiv_id,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    num_pages=0,
                    num_chunks=0,
                    num_tables=0,
                    num_figures=0,
                    num_low_text_pages=0,
                )
            )
            if not config["skip_on_error"]:
                raise

    report = IngestReport(
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        extractor_version=EXTRACTOR_VERSION,
        papers_total=len(statuses),
        papers_ok=sum(1 for s in statuses if s.status == "ok"),
        papers_failed=sum(1 for s in statuses if s.status == "error"),
        total_pages=sum(s.num_pages for s in statuses),
        total_chunks=sum(s.num_chunks for s in statuses),
        tables_total=sum(tables_by_method.values()),
        tables_by_method=tables_by_method,
        figures_total=sum(s.num_figures for s in statuses),
        low_text_layer_pages=sum(s.num_low_text_pages for s in statuses),
        papers=statuses,
    )
    write_ingest_report(report, out_dir)
    return report
