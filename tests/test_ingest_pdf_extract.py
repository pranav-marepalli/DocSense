"""Acceptance criteria 6 and 8: extract_pages, extract_tables, extract_figures."""

import logging
from pathlib import Path

import pymupdf as fitz
import pytest

from docsense.ingest.pdf_extract import (
    extract_figures,
    extract_pages,
    extract_tables,
    extract_title,
)


def test_extract_pages_three_pages_with_low_text_flag(three_page_pdf: Path):
    with fitz.open(three_page_pdf) as doc:
        pages = extract_pages(doc, min_text_chars_per_page=100)

    assert len(pages) == 3
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "Page one" in pages[0].text
    assert "Page three" in pages[2].text

    # near-empty page is flagged but still present
    assert pages[1].low_text_layer is True
    assert pages[1] in pages
    assert pages[0].low_text_layer is False
    assert pages[2].low_text_layer is False


def test_extract_title_from_largest_font_span(tmp_path: Path):
    """R5: no metadata title set, so `extract_title` falls back to the largest-font
    span on page 1. The fixture has two spans of different sizes on page 1, so a
    non-largest-span pick (or a wrong span entirely) would fail this test."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Small print at the bottom of the page.", fontsize=10)
    page.insert_text((72, 120), "The Big Bold Title", fontsize=28)
    path = tmp_path / "largest_span.pdf"
    doc.save(path)
    doc.close()

    with fitz.open(path) as reopened:
        title = reopened.metadata.get("title")
        assert not title  # sanity check: no metadata title set, so the fallback fires
        extracted = extract_title(reopened)

    assert extracted == "The Big Bold Title"


def test_extract_title_prefers_metadata(tmp_path: Path):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Body text")
    doc.set_metadata({"title": "  The Real Title  "})
    path = tmp_path / "titled.pdf"
    doc.save(path)
    doc.close()

    with fitz.open(path) as reopened:
        assert extract_title(reopened) == "The Real Title"


def test_extract_tables_and_figures_on_pdf_with_both(
    pdf_with_table_and_image: Path, tmp_path: Path
):
    image_dir = tmp_path / "images"
    with fitz.open(pdf_with_table_and_image) as doc:
        tables = extract_tables(doc, "2301.00001", caption_radius_pts=40)
        figures = extract_figures(doc, "2301.00001", image_dir, dpi=150, caption_radius_pts=40)

    assert len(figures) >= 1
    figure = figures[0]
    assert figure.page_number == 1
    assert (image_dir / "2301.00001" / "figure_000.png").exists()
    assert figure.image_path == "images/2301.00001/figure_000.png"

    assert len(tables) >= 1
    table = tables[0]
    assert table.method in ("find_tables", "raw_text_fallback")
    assert table.raw_text.strip() != ""

    # R6: the caption-matching contract is actually exercised, not just implemented
    assert table.caption == "Table 1: a test table"
    assert figure.caption == "Figure 1: a test figure"


def test_extract_tables_and_figures_on_plain_pdf_returns_empty(plain_pdf: Path, tmp_path: Path):
    image_dir = tmp_path / "images"
    with fitz.open(plain_pdf) as doc:
        tables = extract_tables(doc, "2301.00002", caption_radius_pts=40)
        figures = extract_figures(doc, "2301.00002", image_dir, dpi=150, caption_radius_pts=40)

    assert tables == []
    assert figures == []


def test_extract_tables_caption_none_outside_tiny_radius(
    pdf_with_table_and_image: Path, tmp_path: Path
):
    """R6: a `caption_radius_pts` too small to reach the real caption text must come
    back `None`, not a mismatch or a raise (Tradeoffs #4: a visible, safe failure)."""
    image_dir = tmp_path / "images"
    with fitz.open(pdf_with_table_and_image) as doc:
        tables = extract_tables(doc, "2301.00001", caption_radius_pts=1.0)
        # sanity check against the same fixture's un-truncated radius (already
        # exercised above, kept here so this test is self-contained)
        extract_figures(doc, "2301.00001", image_dir, dpi=150, caption_radius_pts=40)

    assert tables[0].caption is None


def test_extract_figures_caption_none_outside_tiny_radius(tmp_path: Path):
    """Same contract as above for figures, with a caption placed a known ~20pt gap
    below the image so a tiny radius (1pt) misses it while a normal one (40pt) finds
    it -- the shared fixture's caption overlaps the image bbox, so it can't
    distinguish "found" from "not found" by radius alone."""
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, (0, 0, 10, 10))
    pix.set_rect(pix.irect, (255, 0, 0))
    page.insert_image(fitz.Rect(72, 100, 172, 200), pixmap=pix)
    page.insert_text((72, 232), "Figure 1: a test figure")  # ~20pt below the image
    path = tmp_path / "figure_with_gap.pdf"
    doc.save(path)
    doc.close()

    image_dir = tmp_path / "images"
    with fitz.open(path) as reopened:
        found = extract_figures(reopened, "2301.00003", image_dir, dpi=150, caption_radius_pts=40)
        missed = extract_figures(
            reopened, "2301.00003", image_dir, dpi=150, caption_radius_pts=1.0
        )

    assert found[0].caption == "Figure 1: a test figure"
    assert missed[0].caption is None


# --- R8: per-figure crop/save and detection failures must not abort extraction ---


def test_extract_figures_skips_figure_when_crop_save_fails(
    pdf_with_table_and_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
):
    """A bad raster must not abort extraction (and, transitively, the whole paper):
    `extract_figures` logs a warning and continues past it."""
    image_dir = tmp_path / "images"

    def _boom(self, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(fitz.Page, "get_pixmap", _boom)

    with fitz.open(pdf_with_table_and_image) as doc:
        with caplog.at_level(logging.WARNING):
            figures = extract_figures(doc, "2301.00099", image_dir, dpi=150, caption_radius_pts=40)

    assert figures == []
    assert any("figure" in record.message.lower() for record in caplog.records)


def test_extract_tables_logs_warning_when_find_tables_raises(
    pdf_with_table_and_image: Path, monkeypatch: pytest.MonkeyPatch, caplog
):
    def _boom(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(fitz.Page, "find_tables", _boom)

    with fitz.open(pdf_with_table_and_image) as doc:
        with caplog.at_level(logging.WARNING):
            tables = extract_tables(doc, "2301.00099", caption_radius_pts=40)

    assert tables == []
    assert any("find_tables" in record.message.lower() for record in caplog.records)


def test_extract_figures_logs_warning_when_get_image_info_raises(
    pdf_with_table_and_image: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
):
    def _boom(self, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(fitz.Page, "get_image_info", _boom)

    image_dir = tmp_path / "images"
    with fitz.open(pdf_with_table_and_image) as doc:
        with caplog.at_level(logging.WARNING):
            figures = extract_figures(doc, "2301.00099", image_dir, dpi=150, caption_radius_pts=40)

    assert figures == []
    assert any("get_image_info" in record.message.lower() for record in caplog.records)
