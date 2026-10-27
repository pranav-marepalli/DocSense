"""Pydantic models for the dense index. No I/O here.

`IndexedChunk` is the self-contained sidecar row Phase 2 writes per chunk (Tradeoff
#5: joining `chunks.jsonl` per query is rejected in favor of a self-contained
index directory). `EmbedderInfo` and `IndexManifest` are the "log model and data
versions" record that `retrieve.dense.DenseRetriever.load` checks against config
before querying an index built with a different model.
"""

from typing import Literal

from pydantic import BaseModel


class IndexedChunk(BaseModel):
    """One sidecar row: a Phase 1 Chunk plus its vector row, in row_id order."""

    row_id: int
    chunk_id: str
    arxiv_id: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    token_count: int
    source: Literal["text"]


class EmbedderInfo(BaseModel):
    """Exactly which embedder produced an index; compared against config on load."""

    type: Literal["sentence_transformer", "hash"]
    model_name: str
    dim: int
    normalize: bool
    max_seq_length: int | None
    query_prefix: str
    passage_prefix: str


class IndexManifest(BaseModel):
    """One build, written to <index_dir>/index_manifest.json."""

    variant_name: str
    built_at: str
    seed: int
    embedder: EmbedderInfo
    index_type: str
    num_vectors: int
    num_chunks: int
    num_papers: int
    chunks_path: str
    chunks_sha256: str
    fingerprint: str
    library_versions: dict[str, str]
