"""Acceptance criterion 7: chunk_pages chunk count, overlap, ValueError guard,
and page_start != page_end across a page boundary."""

import pytest

from docsense.ingest.chunk import chunk_pages, count_tokens
from docsense.ingest.schema import Block, Page


def _page(page_number: int, num_words: int, word: str = "w") -> Page:
    text = " ".join(f"{word}{i}" for i in range(num_words))
    return Page(
        page_number=page_number,
        text=text,
        blocks=[Block(bbox=(0.0, 0.0, 1.0, 1.0), text=text)],
        char_count=len(text),
        low_text_layer=False,
    )


def test_count_tokens_whitespace_approximation():
    assert count_tokens("one two three") == 3
    assert count_tokens("") == 0
    assert count_tokens("  leading and trailing  ") == 3


def test_chunk_pages_raises_when_overlap_ge_chunk_size():
    pages = [_page(1, 10)]
    with pytest.raises(ValueError):
        chunk_pages(pages, "2301.00001", chunk_size=50, overlap=50, min_chunk_tokens=10)
    with pytest.raises(ValueError):
        chunk_pages(pages, "2301.00001", chunk_size=50, overlap=60, min_chunk_tokens=10)


def test_chunk_pages_expected_count_and_overlap():
    # 250 tokens on one page, chunk_size=100, overlap=20, min_chunk_tokens=10
    # windows: [0,100), [80,180), [160,250) -> last window has 90 tokens (>= min), no merge.
    pages = [_page(1, 250)]
    chunks = chunk_pages(pages, "2301.00001", chunk_size=100, overlap=20, min_chunk_tokens=10)

    assert len(chunks) == 3
    assert [c.token_count for c in chunks] == [100, 100, 90]

    # verify the overlap: the last 20 tokens of chunk i equal the first 20 of chunk i+1
    for i in range(len(chunks) - 1):
        this_tail = chunks[i].text.split()[-20:]
        next_head = chunks[i + 1].text.split()[:20]
        assert this_tail == next_head


def test_chunk_pages_merges_trailing_chunk_below_minimum():
    # 100 tokens, chunk_size=40, overlap=10 -> windows [0,40) [30,70) [60,100) each 40,40,40
    # tokens total 100 so that's exact. Use 95 tokens to force a short trailing window.
    pages = [_page(1, 95)]
    chunk_size, overlap = 40, 10
    # windows without merge: [0,40) [30,70) [60,95) -> last is 35 tokens, not below min -> no merge
    # use a smaller total to force a <min trailing window instead.
    pages = [_page(1, 85)]
    # windows: [0,40) [30,70) [60,85) -> last is 25 tokens; still above min_chunk_tokens=15? Let's
    # force explicitly below by raising min_chunk_tokens.
    chunks = chunk_pages(
        pages, "2301.00001", chunk_size=chunk_size, overlap=overlap, min_chunk_tokens=30
    )
    # last raw window [60,85) has 25 tokens < min_chunk_tokens(30) -> merged into previous
    assert len(chunks) == 2
    assert chunks[-1].token_count == 85 - 30  # merged window covers [30, 85)


def test_chunk_pages_spans_page_boundary():
    pages = [_page(1, 60), _page(2, 60)]
    chunks = chunk_pages(pages, "2301.00001", chunk_size=80, overlap=10, min_chunk_tokens=10)

    boundary_chunks = [c for c in chunks if c.page_start != c.page_end]
    assert boundary_chunks, "expected at least one chunk spanning the page boundary"
    for c in boundary_chunks:
        assert c.page_start == 1
        assert c.page_end == 2


def test_chunk_pages_ids_and_metadata():
    pages = [_page(1, 50)]
    chunks = chunk_pages(pages, "2301.00001", chunk_size=20, overlap=5, min_chunk_tokens=5)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"2301.00001_chunk_{i:04d}"
        assert chunk.arxiv_id == "2301.00001"
        assert chunk.chunk_index == i
        assert chunk.source == "text"


def test_chunk_pages_empty_pages_returns_empty_list():
    pages = [
        Page(page_number=1, text="", blocks=[], char_count=0, low_text_layer=True),
    ]
    assert chunk_pages(pages, "2301.00001", chunk_size=20, overlap=5, min_chunk_tokens=5) == []
