"""PyMuPDF-based extraction: pages, tables, figures. Each function takes an open
`fitz.Document` (Tradeoffs #1: PyMuPDF over Docling -- already pinned, no model
weights, and `find_tables()`/`get_image_info()` cover tables and figures without
an extra library).
"""

import logging
import re
from pathlib import Path

import pymupdf as fitz

from docsense.ingest.schema import Block, Figure, Page, Table

logger = logging.getLogger(__name__)

TABLE_CAPTION_RE = re.compile(r"^Table\s+\d+", re.IGNORECASE)
FIGURE_CAPTION_RE = re.compile(r"^Fig(ure)?\.?\s+\d+", re.IGNORECASE)


def extract_title(doc: fitz.Document) -> str | None:
    """Metadata title if non-empty, else the largest-font span on page 1, else None."""
    metadata_title = (doc.metadata or {}).get("title")
    if metadata_title and metadata_title.strip():
        return metadata_title.strip()

    if doc.page_count == 0:
        return None

    page = doc[0]
    best_text: str | None = None
    best_size = -1.0
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text and span.get("size", 0) > best_size:
                    best_size = span["size"]
                    best_text = text
    return best_text


def extract_pages(doc: fitz.Document, min_text_chars_per_page: int) -> list[Page]:
    """One Page per PDF page via `page.get_text("blocks")`, `page_number` from 1.

    Sets `low_text_layer` when `char_count < min_text_chars_per_page` and logs a
    warning naming the page; such pages are kept, never dropped (no OCR --
    Tradeoffs #7).
    """
    pages: list[Page] = []
    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_number = page_index + 1
        blocks: list[Block] = []
        texts: list[str] = []
        for raw_block in page.get_text("blocks"):
            x0, y0, x1, y1, text, _block_no, block_type = raw_block[:7]
            if block_type != 0:
                continue
            text = text.strip()
            if not text:
                continue
            blocks.append(Block(bbox=(x0, y0, x1, y1), text=text))
            texts.append(text)

        page_text = "\n".join(texts)
        char_count = len(page_text)
        low_text_layer = char_count < min_text_chars_per_page
        if low_text_layer:
            logger.warning(
                "page %d has a low text layer (%d chars < threshold %d)",
                page_number,
                char_count,
                min_text_chars_per_page,
            )
        pages.append(
            Page(
                page_number=page_number,
                text=page_text,
                blocks=blocks,
                char_count=char_count,
                low_text_layer=low_text_layer,
            )
        )
    return pages


def _find_caption(
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
    caption_radius_pts: float,
    pattern: re.Pattern[str],
) -> str | None:
    """Nearest text block matching `pattern` within `caption_radius_pts` above or below `bbox`."""
    _bx0, by0, _bx1, by1 = bbox
    best: str | None = None
    best_dist: float | None = None
    for raw_block in page.get_text("blocks"):
        _tx0, ty0, _tx1, ty1, text = raw_block[:5]
        candidate = text.strip()
        if not pattern.match(candidate):
            continue
        if ty1 <= by0:
            distance = by0 - ty1
        elif ty0 >= by1:
            distance = ty0 - by1
        else:
            distance = 0.0
        if distance <= caption_radius_pts and (best_dist is None or distance < best_dist):
            best = candidate
            best_dist = distance
    return best


def extract_tables(doc: fitz.Document, arxiv_id: str, caption_radius_pts: float) -> list[Table]:
    """`page.find_tables()` per page.

    Per hit, `Table.to_markdown()` with `method="find_tables"`, else (on raise
    or empty) `get_text(clip=bbox)` with `markdown=None` and
    `method="raw_text_fallback"`. `raw_text` always filled. Caption = nearest
    block matching `^Table\\s+\\d+` within `caption_radius_pts` above or below
    the bbox.
    """
    tables: list[Table] = []
    table_index = 0
    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_number = page_index + 1
        try:
            hits = page.find_tables().tables
        except Exception:
            logger.warning(
                "find_tables failed on page %d of %s; treating as no tables",
                page_number,
                arxiv_id,
                exc_info=True,
            )
            hits = []

        for hit in hits:
            bbox = tuple(hit.bbox)
            raw_text = page.get_text(clip=bbox)
            try:
                markdown = hit.to_markdown()
                if not markdown or not markdown.strip():
                    raise ValueError("empty markdown")
                method: str = "find_tables"
            except Exception:
                markdown = None
                method = "raw_text_fallback"

            caption = _find_caption(page, bbox, caption_radius_pts, TABLE_CAPTION_RE)
            tables.append(
                Table(
                    table_id=f"{arxiv_id}_table_{table_index:03d}",
                    page_number=page_number,
                    bbox=bbox,
                    caption=caption,
                    markdown=markdown,
                    raw_text=raw_text,
                    method=method,  # type: ignore[arg-type]
                )
            )
            table_index += 1
    return tables


def extract_figures(
    doc: fitz.Document,
    arxiv_id: str,
    image_dir: Path,
    dpi: int,
    caption_radius_pts: float,
) -> list[Figure]:
    """`page.get_image_info(xrefs=True)` to locate embedded rasters.

    Crops via `page.get_pixmap(clip=bbox, dpi=dpi)` and writes a PNG under
    `image_dir/<arxiv_id>/`. Caption lookup mirrors `extract_tables` with
    `^Fig(ure)?\\.?\\s+\\d+`.
    """
    figures: list[Figure] = []
    figure_index = 0
    paper_image_dir = image_dir / arxiv_id
    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_number = page_index + 1
        try:
            infos = page.get_image_info(xrefs=True)
        except Exception:
            logger.warning(
                "get_image_info failed on page %d of %s; treating as no figures",
                page_number,
                arxiv_id,
                exc_info=True,
            )
            infos = []

        for info in infos:
            bbox = tuple(info["bbox"])
            paper_image_dir.mkdir(parents=True, exist_ok=True)
            filename = f"figure_{figure_index:03d}.png"
            dest = paper_image_dir / filename
            try:
                pixmap = page.get_pixmap(clip=bbox, dpi=dpi)
                pixmap.save(dest)
            except Exception:
                logger.warning(
                    "failed to crop/save figure %d on page %d of %s; skipping this figure",
                    figure_index,
                    page_number,
                    arxiv_id,
                    exc_info=True,
                )
                continue

            caption = _find_caption(page, bbox, caption_radius_pts, FIGURE_CAPTION_RE)
            image_path = Path(image_dir.name) / arxiv_id / filename
            figures.append(
                Figure(
                    figure_id=f"{arxiv_id}_figure_{figure_index:03d}",
                    page_number=page_number,
                    bbox=bbox,
                    caption=caption,
                    image_path=image_path.as_posix(),
                )
            )
            figure_index += 1
    return figures
