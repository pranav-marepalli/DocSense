"""The page-level citation contract, retrieval side.

An arXiv id or page number in a DocSense citation is always carried from the
Chunk that Phase 1 extracted, never inferred -- so every function here is pure
field copying (Tradeoff #6).
"""

from docsense.retrieve.schema import Citation, RetrievalResult, RetrievedChunk


def citation_for(chunk: RetrievedChunk) -> Citation:
    """Project one hit onto its provenance."""
    return Citation(
        chunk_id=chunk.chunk_id,
        arxiv_id=chunk.arxiv_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        rank=chunk.rank,
        score=chunk.score,
    )


def citations_for(result: RetrievalResult) -> list[Citation]:
    """Citations for every hit, in rank order."""
    return [citation_for(hit) for hit in result.results]


def format_citation(citation: Citation) -> str:
    """Human/log rendering: "arXiv:2301.12345 p.4-5", collapsing to "p.4" when
    page_start == page_end."""
    if citation.page_start == citation.page_end:
        pages = f"p.{citation.page_start}"
    else:
        pages = f"p.{citation.page_start}-{citation.page_end}"
    return f"arXiv:{citation.arxiv_id} {pages}"
