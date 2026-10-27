"""Pydantic models for retrieval. No I/O here.

`RetrievedChunk` carries citation fields verbatim from the index (nothing
downstream re-derives page numbers from text). `Citation` is a `RetrievedChunk`
reduced to its provenance -- what Phase 3 scores and Phase 6 renders.
`RetrievalResult` is one query's hits plus the provenance to reproduce them.
"""

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """One hit, carrying its citation fields verbatim."""

    chunk_id: str
    arxiv_id: str
    page_start: int
    page_end: int
    text: str
    score: float
    rank: int


class Citation(BaseModel):
    """A RetrievedChunk reduced to its provenance. Every field is copied, never inferred."""

    chunk_id: str
    arxiv_id: str
    page_start: int
    page_end: int
    rank: int
    score: float


class RetrievalResult(BaseModel):
    """One query's hits plus the provenance to reproduce them."""

    query: str
    retriever: str
    top_k: int
    variant_name: str
    embedder_model: str
    index_fingerprint: str
    latency_ms: float
    """Mean per-query latency (encode + search), measured over the whole
    `retrieve_batch` call and divided by the number of queries in that batch --
    not a true per-query wall-clock measurement. For a single `retrieve()` call
    (a batch of one), it is the true latency."""
    results: list[RetrievedChunk]
