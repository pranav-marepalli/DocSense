"""Turn a paper's pages into retrieval-sized, page-cited chunks.

Operates purely on `list[Page]`; independent of PDFs so it is unit-testable
without fitz. Fixed-size windowing over whitespace-approximated tokens
(Tradeoffs #3): deterministic, dependency-free, and avoids tying ingestion to
one embedder's tokenizer when the embedder varies by ablation config.
"""

from docsense.ingest.schema import Chunk, Page


def count_tokens(text: str) -> int:
    """Approximate token count via whitespace splitting (Tradeoffs #3)."""
    return len(text.split())


def chunk_pages(
    pages: list[Page],
    arxiv_id: str,
    chunk_size: int,
    overlap: int,
    min_chunk_tokens: int,
) -> list[Chunk]:
    """Slide a `chunk_size`-token window with `overlap` tokens of overlap over `pages`.

    Concatenates page text in order while tracking which page each token came
    from. `page_start`/`page_end` are the first/last page any token in the
    chunk came from. A trailing chunk below `min_chunk_tokens` is merged into
    the previous chunk rather than dropped. Low-text-layer pages contribute
    whatever text they have.

    Raises:
        ValueError: if `overlap >= chunk_size`.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")

    tokens: list[str] = []
    token_pages: list[int] = []
    for page in pages:
        page_tokens = page.text.split()
        tokens.extend(page_tokens)
        token_pages.extend([page.page_number] * len(page_tokens))

    if not tokens:
        return []

    windows: list[tuple[int, int]] = []
    start = 0
    step = chunk_size - overlap
    n = len(tokens)
    while start < n:
        end = min(start + chunk_size, n)
        windows.append((start, end))
        if end == n:
            break
        start += step

    # Merge a trailing window below min_chunk_tokens into the previous one.
    if len(windows) > 1:
        last_start, last_end = windows[-1]
        if (last_end - last_start) < min_chunk_tokens:
            windows.pop()
            prev_start, _ = windows[-1]
            windows[-1] = (prev_start, last_end)

    chunks: list[Chunk] = []
    for i, (start, end) in enumerate(windows):
        chunk_tokens = tokens[start:end]
        chunk_token_pages = token_pages[start:end]
        text = " ".join(chunk_tokens)
        chunks.append(
            Chunk(
                chunk_id=f"{arxiv_id}_chunk_{i:04d}",
                arxiv_id=arxiv_id,
                chunk_index=i,
                page_start=min(chunk_token_pages),
                page_end=max(chunk_token_pages),
                text=text,
                token_count=len(chunk_tokens),
                source="text",
            )
        )
    return chunks
