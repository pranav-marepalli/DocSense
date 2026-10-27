"""Acceptance criterion 1 (index half): every model in index/schema.py
round-trips model_dump_json() -> model_validate_json() to an equal object."""

from docsense.index.schema import EmbedderInfo, IndexedChunk, IndexManifest


def _roundtrip(model):
    cls = type(model)
    return cls.model_validate_json(model.model_dump_json())


def test_indexed_chunk_roundtrips():
    model = IndexedChunk(
        row_id=0,
        chunk_id="2301.00001_chunk_0000",
        arxiv_id="2301.00001",
        chunk_index=0,
        page_start=1,
        page_end=1,
        text="some chunk text",
        token_count=3,
        source="text",
    )
    assert _roundtrip(model) == model


def test_embedder_info_roundtrips():
    model = EmbedderInfo(
        type="sentence_transformer",
        model_name="BAAI/bge-small-en-v1.5",
        dim=384,
        normalize=True,
        max_seq_length=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
        passage_prefix="",
    )
    assert _roundtrip(model) == model


def test_embedder_info_roundtrips_with_null_max_seq_length():
    model = EmbedderInfo(
        type="hash",
        model_name="hash",
        dim=16,
        normalize=True,
        max_seq_length=None,
        query_prefix="",
        passage_prefix="",
    )
    assert _roundtrip(model) == model


def test_index_manifest_roundtrips():
    model = IndexManifest(
        variant_name="baseline",
        built_at="2024-01-01T00:00:00+00:00",
        seed=0,
        embedder=EmbedderInfo(
            type="hash",
            model_name="hash",
            dim=16,
            normalize=True,
            max_seq_length=None,
            query_prefix="",
            passage_prefix="",
        ),
        index_type="numpy_flat_ip",
        num_vectors=12,
        num_chunks=12,
        num_papers=3,
        chunks_path="data/parsed/chunks.jsonl",
        chunks_sha256="a" * 64,
        fingerprint="b" * 64,
        library_versions={"numpy": "2.0.0", "torch": "2.0.0", "faiss": "1.8.0"},
    )
    assert _roundtrip(model) == model
