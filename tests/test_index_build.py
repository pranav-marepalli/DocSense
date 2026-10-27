"""Acceptance criteria 5, 6, 7, 8: build_index output shape, citation-field
preservation, idempotency/fingerprinting, and error handling."""

from pathlib import Path

import pytest

from docsense.index.build import build_index
from docsense.index.schema import IndexManifest
from docsense.ingest.io import get_record_store, read_chunks
from tests.conftest import write_sample_chunks


def _config(tmp_path: Path) -> dict:
    return {
        "variant_name": "test",
        "seed": 0,
        "record_format": "jsonl",
        "embedder": {
            "type": "hash",
            "model_name": "BAAI/bge-small-en-v1.5",
            "dim": 16,
            "normalize": True,
            "batch_size": 4,
            "device": None,
            "max_seq_length": 512,
            "query_prefix": "",
            "passage_prefix": "",
            "trust_remote_code": False,
        },
        "index": {"type": "numpy_flat_ip", "num_threads": 1},
        "retrieval": {"top_k": 5, "max_chunks_per_paper": None},
    }


def test_build_index_writes_expected_files_and_manifest(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = write_sample_chunks(chunks_path)
    index_dir = tmp_path / "index"

    manifest = build_index(chunks_path, index_dir, _config(tmp_path))

    assert sorted(p.name for p in index_dir.iterdir()) == [
        "chunk_meta.jsonl",
        "index_manifest.json",
        "vectors.npy",
    ]

    store = get_record_store("jsonl")
    meta_records = store.read_all(index_dir / "chunk_meta.jsonl")
    assert [r["row_id"] for r in meta_records] == list(range(len(chunks)))
    for i, record in enumerate(meta_records):
        assert record["chunk_id"] == chunks[i].chunk_id

    assert manifest.num_vectors == manifest.num_chunks == len(chunks)
    assert manifest.num_papers == len({c.arxiv_id for c in chunks})
    assert manifest.variant_name == "test"
    assert manifest.seed == 0
    assert manifest.embedder.type == "hash"
    assert manifest.index_type == "numpy_flat_ip"
    assert len(manifest.chunks_sha256) == 64
    assert len(manifest.fingerprint) == 64
    assert manifest.library_versions["numpy"]
    assert manifest.library_versions["torch"]
    assert manifest.library_versions["faiss"]
    assert manifest.library_versions["sentence_transformers"]


def test_indexed_chunk_preserves_citation_fields(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = write_sample_chunks(chunks_path)
    index_dir = tmp_path / "index"

    build_index(chunks_path, index_dir, _config(tmp_path))

    store = get_record_store("jsonl")
    meta_records = store.read_all(index_dir / "chunk_meta.jsonl")
    for chunk, record in zip(chunks, meta_records, strict=True):
        assert record["chunk_id"] == chunk.chunk_id
        assert record["arxiv_id"] == chunk.arxiv_id
        assert record["page_start"] == chunk.page_start
        assert record["page_end"] == chunk.page_end
        assert record["text"] == chunk.text


def test_build_index_is_idempotent_until_force_or_input_change(tmp_path: Path, monkeypatch):
    chunks_path = tmp_path / "chunks.jsonl"
    write_sample_chunks(chunks_path)
    index_dir = tmp_path / "index"
    config = _config(tmp_path)

    first = build_index(chunks_path, index_dir, config)

    encode_calls = []
    from docsense.index.embed import HashEmbedder

    original_encode = HashEmbedder.encode

    def _counting_encode(self, texts, is_query=False):
        encode_calls.append(texts)
        return original_encode(self, texts, is_query=is_query)

    monkeypatch.setattr(HashEmbedder, "encode", _counting_encode)

    second = build_index(chunks_path, index_dir, config)
    assert second.fingerprint == first.fingerprint
    assert encode_calls == []

    third = build_index(chunks_path, index_dir, config, force=True)
    assert encode_calls != []
    assert third.fingerprint == first.fingerprint

    encode_calls.clear()
    with chunks_path.open("a") as f:
        f.write(
            '{"chunk_id": "2301.00099_chunk_0000", "arxiv_id": "2301.00099", '
            '"chunk_index": 0, "page_start": 1, "page_end": 1, '
            '"text": "brand new extra chunk text", "token_count": 5, "source": "text"}\n'
        )
    fourth = build_index(chunks_path, index_dir, config)
    assert fourth.fingerprint != first.fingerprint
    assert encode_calls != []
    assert fourth.num_chunks == first.num_chunks + 1


def test_build_index_missing_chunks_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path / "does_not_exist.jsonl", tmp_path / "index", _config(tmp_path))
    assert not (tmp_path / "index").exists()


def test_build_index_empty_chunks_path_raises(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("")
    index_dir = tmp_path / "index"

    with pytest.raises(ValueError):
        build_index(chunks_path, index_dir, _config(tmp_path))
    assert not index_dir.exists()


def test_build_index_manifest_is_readable_back(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    write_sample_chunks(chunks_path)
    index_dir = tmp_path / "index"

    build_index(chunks_path, index_dir, _config(tmp_path))

    reloaded = IndexManifest.model_validate_json((index_dir / "index_manifest.json").read_text())
    assert reloaded.num_chunks == len(read_chunks(chunks_path, get_record_store("jsonl")))
