"""Builds a self-contained dense index directory from `data/parsed/chunks.jsonl`.

Reads chunks via `ingest.io.read_chunks` (one source of truth for the Chunk
schema), embeds them as passages, adds them to a `VectorStore` in file order so
`row_id == line number`, and writes `chunk_meta.jsonl` plus `index_manifest.json`
-- the "log model and data versions" record for retrieval (Tradeoff #4:
fingerprint-based idempotency over a per-chunk cache).
"""

import hashlib
import json
import os
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import sentence_transformers
import torch

from docsense.index.embed import get_embedder
from docsense.index.schema import EmbedderInfo, IndexedChunk, IndexManifest
from docsense.index.store import get_vector_store
from docsense.ingest.io import RecordStore, get_record_store, read_chunks

# src/docsense/index/build.py -> repo root (same pattern as docsense.config.CONFIG_DIR).
REPO_ROOT = Path(__file__).resolve().parents[3]


def _relative_to_repo_root(path: Path) -> str:
    """Best-effort path relative to `REPO_ROOT` (may contain `..` if `path` is outside it)."""
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    """sha256 of `path`'s bytes -- the input corpus identity in the manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_meta_filename(record_store: RecordStore) -> str:
    """Sidecar filename for `record_store`'s format, e.g. `"chunk_meta.jsonl"`.

    The one source of truth for this name: `build_index` writes it and
    `DenseRetriever.load` reads it, so both must agree without either
    hardcoding the extension.
    """
    return f"chunk_meta.{record_store.extension}"


def corpus_fingerprint(
    embedder_info: EmbedderInfo, index_type: str, chunks_sha256: str, num_chunks: int
) -> str:
    """sha256 over the canonical JSON of its arguments -- the rebuild-needed test."""
    payload = {
        "embedder": embedder_info.model_dump(mode="json"),
        "index_type": index_type,
        "chunks_sha256": chunks_sha256,
        "num_chunks": num_chunks,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _faiss_version() -> str:
    """`faiss-cpu`'s installed version without importing faiss (Tradeoff #2: a
    faiss-less environment must still be able to import this module)."""
    try:
        return version("faiss-cpu")
    except PackageNotFoundError:
        return "not installed"


def _library_versions() -> dict[str, str]:
    """Versions logged into every manifest, per CLAUDE.md's reproducibility convention."""
    return {
        "sentence_transformers": sentence_transformers.__version__,
        "faiss": _faiss_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def build_index(
    chunks_path: Path, index_dir: Path, config: dict, force: bool = False
) -> IndexManifest:
    """Embed every chunk in `chunks_path` and write a self-contained index directory.

    `<index_dir>/{vectors.*, chunk_meta.jsonl, index_manifest.json}`. Idempotent:
    if `index_manifest.json` exists and its `fingerprint` matches the current
    inputs, returns it without re-embedding unless `force`.

    Raises:
        FileNotFoundError: if chunks_path does not exist.
        ValueError: if chunks_path holds no chunks (an empty index is never written).
    """
    if not chunks_path.is_file():
        raise FileNotFoundError(f"No such chunks file: {chunks_path}")

    record_store = get_record_store(config["record_format"])
    chunks = read_chunks(chunks_path, record_store)
    if not chunks:
        raise ValueError(f"{chunks_path} holds no chunks; an empty index is never written")

    chunks_sha256 = _sha256_file(chunks_path)
    embedder = get_embedder(config)
    embedder_info = embedder.info
    index_type = config["index"]["type"]
    num_threads = config["index"]["num_threads"]
    fingerprint = corpus_fingerprint(embedder_info, index_type, chunks_sha256, len(chunks))

    manifest_path = index_dir / "index_manifest.json"
    if not force and manifest_path.is_file():
        existing = IndexManifest.model_validate_json(manifest_path.read_text())
        if existing.fingerprint == fingerprint:
            return existing

    batch_size = config["embedder"]["batch_size"]
    store = get_vector_store(index_type, embedder_info.dim, num_threads=num_threads)
    chunk_meta: list[IndexedChunk] = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.encode([chunk.text for chunk in batch], is_query=False)
        store.add(vectors)
        for offset, chunk in enumerate(batch):
            chunk_meta.append(
                IndexedChunk(
                    row_id=start + offset,
                    chunk_id=chunk.chunk_id,
                    arxiv_id=chunk.arxiv_id,
                    chunk_index=chunk.chunk_index,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    source=chunk.source,
                )
            )

    index_dir.mkdir(parents=True, exist_ok=True)
    store.save(index_dir)

    meta_path = index_dir / chunk_meta_filename(record_store)
    if meta_path.exists():
        meta_path.unlink()
    record_store.append((chunk.model_dump(mode="json") for chunk in chunk_meta), meta_path)

    manifest = IndexManifest(
        variant_name=config["variant_name"],
        built_at=datetime.now(UTC).isoformat(),
        seed=config["seed"],
        embedder=embedder_info,
        index_type=index_type,
        num_vectors=len(store),
        num_chunks=len(chunks),
        num_papers=len({chunk.arxiv_id for chunk in chunks}),
        chunks_path=_relative_to_repo_root(chunks_path),
        chunks_sha256=chunks_sha256,
        fingerprint=fingerprint,
        library_versions=_library_versions(),
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest
