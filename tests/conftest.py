"""Shared pytest fixtures for DocSense tests."""

from pathlib import Path

import pymupdf as fitz
import pytest

from docsense.ingest.io import get_record_store, write_chunks
from docsense.ingest.schema import Chunk

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Distinct enough bag-of-words per chunk that HashEmbedder (sha256 hashing, no
# semantics) still tells every chunk apart -- Phase 2 fixtures need that so a
# chunk's own text, re-embedded as a query, is unambiguously its own nearest hit.
_SAMPLE_PAPERS = {
    "2301.00001": [
        "alpha bravo charlie delta echo foxtrot golf hotel",
        "india juliet kilo lima mike november oscar papa",
        "quebec romeo sierra tango uniform victor whiskey xray",
        "yankee zulu amber birch cedar dawn elm fern",
        "grove holly ivy jade kelp larch maple nectar",
    ],
    "2301.00002": [
        "opal pine quartz rowan sage thistle umber vine",
        "willow xenon yarrow zinnia acorn basil clover daisy",
        "ember fennel ginger heather iris jasmine kudzu lotus",
        "marigold nutmeg orchid poppy quince rue saffron tulip",
    ],
    "2301.00003": [
        "almond bamboo cactus date elder fig guava hazel",
        "indigo jackfruit kiwano lychee mango nance olive papaya",
        "quandong raspberry starfruit tamarind ugli vanilla walnut",
    ],
}


def write_sample_chunks(path: Path) -> list[Chunk]:
    """Write a fixture chunks.jsonl: 3 papers, 12 chunks total, distinct
    bag-of-words text per chunk. Returns the Chunk objects written, in file order."""
    store = get_record_store("jsonl")
    chunks: list[Chunk] = []
    for arxiv_id, bodies in _SAMPLE_PAPERS.items():
        for chunk_index, body in enumerate(bodies):
            page = chunk_index + 1
            chunks.append(
                Chunk(
                    chunk_id=f"{arxiv_id}_chunk_{chunk_index:04d}",
                    arxiv_id=arxiv_id,
                    chunk_index=chunk_index,
                    page_start=page,
                    page_end=page,
                    text=body,
                    token_count=len(body.split()),
                    source="text",
                )
            )
    write_chunks(chunks, path, store)
    return chunks


@pytest.fixture(autouse=True)
def _block_real_sentence_transformer(monkeypatch: pytest.MonkeyPatch):
    """Acceptance criterion 16: no test may construct a real SentenceTransformer
    (that would download model weights / make a network request). Tests that
    exercise SentenceTransformerEmbedder must monkeypatch
    `docsense.index.embed.SentenceTransformer` with a fake themselves."""
    from docsense.index import embed as embed_module

    def _blocked(*args, **kwargs):
        raise AssertionError(
            "a test attempted to construct a real SentenceTransformer; monkeypatch "
            "docsense.index.embed.SentenceTransformer with a fake instead"
        )

    monkeypatch.setattr(embed_module, "SentenceTransformer", _blocked)


@pytest.fixture
def three_page_pdf(tmp_path: Path) -> Path:
    """A 3-page PDF: pages 1 and 3 have plenty of text, page 2 is near-empty."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Page one has plenty of text. " * 10)
    doc.new_page()  # page 2: intentionally near-empty
    page3 = doc.new_page()
    page3.insert_text((72, 72), "Page three has plenty of text. " * 10)

    path = tmp_path / "three_page.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def pdf_with_table_and_image(tmp_path: Path) -> Path:
    """A 1-page PDF with one embedded image and one ruled-line table-like grid."""
    doc = fitz.open()
    page = doc.new_page()

    page.insert_text((72, 72), "A paper with a table and a figure.")

    # table: 3x3 ruled grid with a caption above it
    x0, y0, w, h = 72, 150, 300, 90
    rows, cols = 3, 3
    for r in range(rows + 1):
        page.draw_line((x0, y0 + r * h / rows), (x0 + w, y0 + r * h / rows))
    for c in range(cols + 1):
        page.draw_line((x0 + c * w / cols, y0), (x0 + c * w / cols, y0 + h))
    labels = [["A", "B", "C"], ["1", "2", "3"], ["4", "5", "6"]]
    for r in range(rows):
        for c in range(cols):
            page.insert_text((x0 + c * w / cols + 5, y0 + r * h / rows + 15), labels[r][c])
    page.insert_text((x0, y0 - 20), "Table 1: a test table")

    # figure: a small embedded raster with a caption below it
    pix = fitz.Pixmap(fitz.csRGB, (0, 0, 10, 10))
    pix.set_rect(pix.irect, (255, 0, 0))
    image_rect = fitz.Rect(72, 300, 172, 400)
    page.insert_image(image_rect, pixmap=pix)
    page.insert_text((72, 410), "Figure 1: a test figure")

    path = tmp_path / "with_table_and_image.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def plain_pdf(tmp_path: Path) -> Path:
    """A 1-page PDF with only prose text: no tables, no images."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Just some plain prose text, nothing else here.")
    path = tmp_path / "plain.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def truncated_pdf(tmp_path: Path) -> Path:
    """A file with a .pdf extension that is not a valid PDF, to trigger an extraction error."""
    path = tmp_path / "truncated.pdf"
    path.write_bytes(b"%PDF-1.4 not actually a valid pdf body")
    return path
