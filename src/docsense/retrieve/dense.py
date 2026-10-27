"""The baseline (non-hybrid, non-reranked) retriever.

`DenseRetriever` embeds a query, runs exact top-k inner product over a loaded
`VectorStore`, and maps row_ids back to `IndexedChunk`s. Hybrid retrieval and
reranking are Phase 4.
"""

import time
from pathlib import Path

import numpy as np

from docsense.index.build import chunk_meta_filename
from docsense.index.embed import Embedder, get_embedder
from docsense.index.schema import IndexedChunk, IndexManifest
from docsense.index.store import VectorStore, load_vector_store
from docsense.ingest.io import get_record_store
from docsense.retrieve.schema import RetrievalResult, RetrievedChunk

_MISMATCH_FIELDS = ("model_name", "dim", "normalize", "query_prefix", "passage_prefix")


def _validate_chunk_meta(
    chunk_meta: list[IndexedChunk], store: VectorStore, manifest: IndexManifest
) -> None:
    """Guard against a tampered or stale sidecar.

    `chunk_meta[row_id]` is a list-indexing lookup, so a `chunk_meta.jsonl` that
    is reordered, truncated, or duplicated would otherwise yield silently wrong
    citations with no exception.

    Raises:
        ValueError: if `len(chunk_meta)` disagrees with `len(store)` or
            `manifest.num_vectors`, or if `chunk_meta`'s `row_id`s are not
            exactly `0..len(chunk_meta)-1` in ascending order.
    """
    if not (len(chunk_meta) == len(store) == manifest.num_vectors):
        raise ValueError(
            "chunk_meta sidecar is inconsistent with the index: "
            f"len(chunk_meta)={len(chunk_meta)}, len(store)={len(store)}, "
            f"manifest.num_vectors={manifest.num_vectors}; the sidecar may be "
            "truncated, duplicated, or stale"
        )
    expected_row_ids = list(range(len(chunk_meta)))
    actual_row_ids = [chunk.row_id for chunk in chunk_meta]
    if actual_row_ids != expected_row_ids:
        raise ValueError(
            "chunk_meta sidecar row_ids are not 0..N-1 in ascending order "
            f"(got {actual_row_ids[:10]}{'...' if len(actual_row_ids) > 10 else ''}); "
            "the sidecar may be reordered, truncated, or have duplicate rows"
        )


class DenseRetriever:
    """Embed the query, exact top-k inner product, map row_ids back to IndexedChunks.

    No hybrid, no reranking (Phase 4).
    """

    name: str = "dense"

    def __init__(
        self,
        store: VectorStore,
        chunk_meta: list[IndexedChunk],
        embedder: Embedder,
        manifest: IndexManifest,
        top_k: int,
        max_chunks_per_paper: int | None = None,
    ) -> None:
        self._store = store
        self._chunk_meta = chunk_meta
        self._embedder = embedder
        self._manifest = manifest
        self._top_k = top_k
        self._max_chunks_per_paper = max_chunks_per_paper

    @classmethod
    def load(cls, index_dir: Path, config: dict) -> "DenseRetriever":
        """Load manifest, store, and chunk_meta from `index_dir`; build the embedder from config.

        Raises:
            FileNotFoundError: if `index_dir` has no manifest.
            ValueError: if the manifest's EmbedderInfo disagrees with config's
                (model_name, dim, normalize, prefixes) -- querying an index with a
                model other than the one that built it is silently wrong, so it is
                a hard error. Also raised if `chunk_meta`'s row count disagrees with
                the vector store or manifest, or its `row_id`s are not exactly
                `0..len(chunk_meta)-1` in order -- row_ids are resolved into
                `chunk_meta` by list indexing, so a reordered, truncated, or
                duplicated sidecar must fail loudly rather than yield silently
                wrong citations.
        """
        manifest_path = index_dir / "index_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No index manifest at {manifest_path}")
        manifest = IndexManifest.model_validate_json(manifest_path.read_text())

        embedder = get_embedder(config)
        configured = embedder.info
        built = manifest.embedder
        for field in _MISMATCH_FIELDS:
            if getattr(configured, field) != getattr(built, field):
                raise ValueError(
                    f"embedder.{field} in config ({getattr(configured, field)!r}) does not match "
                    f"the index's ({getattr(built, field)!r}); querying an index with a different "
                    "model than the one that built it is unsupported"
                )

        store = load_vector_store(
            index_dir,
            manifest.index_type,
            manifest.embedder.dim,
            num_threads=config["index"]["num_threads"],
        )
        record_store = get_record_store(config["record_format"])
        chunk_meta = [
            IndexedChunk.model_validate(record)
            for record in record_store.read_all(index_dir / chunk_meta_filename(record_store))
        ]
        _validate_chunk_meta(chunk_meta, store, manifest)

        return cls(
            store=store,
            chunk_meta=chunk_meta,
            embedder=embedder,
            manifest=manifest,
            top_k=config["retrieval"]["top_k"],
            max_chunks_per_paper=config["retrieval"]["max_chunks_per_paper"],
        )

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Ranked hits for `query`, `top_k` defaulting to the configured value.

        Raises:
            ValueError: on an empty or whitespace-only query.
        """
        return self.retrieve_batch([query], top_k=top_k)[0]

    def retrieve_batch(self, queries: list[str], top_k: int | None = None) -> list[RetrievalResult]:
        """One embedding call for all queries, same per-query semantics as `retrieve`."""
        if not queries:
            return []
        for query in queries:
            if not query.strip():
                raise ValueError("query must not be empty or whitespace-only")

        k = top_k if top_k is not None else self._top_k
        search_k = len(self._chunk_meta) if self._max_chunks_per_paper is not None else k

        started = time.perf_counter()
        vectors = self._embedder.encode(queries, is_query=True)
        scores, row_ids = self._store.search(vectors, search_k)
        batch_latency_ms = (time.perf_counter() - started) * 1000.0
        # Mean per-query latency within this batch call, not a true per-query
        # wall-clock measurement (see RetrievalResult.latency_ms docstring).
        latency_ms = batch_latency_ms / len(queries)

        return [
            RetrievalResult(
                query=query,
                retriever=self.name,
                top_k=k,
                variant_name=self._manifest.variant_name,
                embedder_model=self._manifest.embedder.model_name,
                index_fingerprint=self._manifest.fingerprint,
                latency_ms=latency_ms,
                results=self._hits(scores[i], row_ids[i], k),
            )
            for i, query in enumerate(queries)
        ]

    def _hits(self, scores: np.ndarray, row_ids: np.ndarray, k: int) -> list[RetrievedChunk]:
        """Build ranked hits from one query's (scores, row_ids), applying the per-paper cap."""
        hits: list[RetrievedChunk] = []
        per_paper_count: dict[str, int] = {}
        for score, row_id in zip(scores, row_ids, strict=True):
            if row_id < 0:
                continue
            meta = self._chunk_meta[int(row_id)]
            if self._max_chunks_per_paper is not None:
                count = per_paper_count.get(meta.arxiv_id, 0)
                if count >= self._max_chunks_per_paper:
                    continue
                per_paper_count[meta.arxiv_id] = count + 1
            hits.append(
                RetrievedChunk(
                    chunk_id=meta.chunk_id,
                    arxiv_id=meta.arxiv_id,
                    page_start=meta.page_start,
                    page_end=meta.page_end,
                    text=meta.text,
                    score=float(score),
                    rank=len(hits),
                )
            )
            if len(hits) >= k:
                break
        return hits
