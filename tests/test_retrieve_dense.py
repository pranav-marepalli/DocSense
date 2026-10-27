"""Acceptance criteria 9, 10, 11, 12: DenseRetriever.load/retrieve/retrieve_batch
invariants, embedder-mismatch guard, and the per-paper cap."""

import json
from pathlib import Path

import pytest

from docsense.index.build import build_index
from docsense.index.embed import HashEmbedder
from docsense.retrieve.dense import DenseRetriever
from tests.conftest import write_sample_chunks


def _config(
    embedder_overrides: dict | None = None,
    retrieval_overrides: dict | None = None,
    index_overrides: dict | None = None,
) -> dict:
    embedder = {
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
    }
    embedder.update(embedder_overrides or {})
    retrieval = {"top_k": 5, "max_chunks_per_paper": None}
    retrieval.update(retrieval_overrides or {})
    index = {"type": "numpy_flat_ip", "num_threads": 1}
    index.update(index_overrides or {})
    return {
        "variant_name": "test",
        "seed": 0,
        "record_format": "jsonl",
        "embedder": embedder,
        "index": index,
        "retrieval": retrieval,
    }


@pytest.fixture
def built_index(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = write_sample_chunks(chunks_path)
    index_dir = tmp_path / "index"
    config = _config()
    build_index(chunks_path, index_dir, config)
    return index_dir, config, chunks


def test_retrieve_own_text_ranks_first(built_index):
    index_dir, config, chunks = built_index
    retriever = DenseRetriever.load(index_dir, config)

    target = chunks[5]
    result = retriever.retrieve(target.text)

    assert result.results[0].rank == 0
    assert result.results[0].chunk_id == target.chunk_id
    assert result.results[0].score == max(hit.score for hit in result.results)


def test_retrieval_result_invariants(built_index):
    index_dir, config, chunks = built_index
    retriever = DenseRetriever.load(index_dir, config)

    result = retriever.retrieve(chunks[0].text, top_k=3)

    assert len(result.results) <= 3
    assert result.top_k == 3
    assert [hit.rank for hit in result.results] == list(range(len(result.results)))
    scores = [hit.score for hit in result.results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    assert result.retriever == "dense"
    assert result.variant_name == "test"
    assert result.embedder_model == "hash"

    from docsense.index.schema import IndexManifest

    manifest = IndexManifest.model_validate_json((index_dir / "index_manifest.json").read_text())
    assert result.index_fingerprint == manifest.fingerprint


def test_retrieve_empty_or_whitespace_query_raises(built_index):
    index_dir, config, _ = built_index
    retriever = DenseRetriever.load(index_dir, config)

    with pytest.raises(ValueError):
        retriever.retrieve("")
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "a-different-model"),
        ("dim", 999),
        ("normalize", False),
        ("query_prefix", "a different query prefix: "),
        ("passage_prefix", "a different passage prefix: "),
    ],
)
def test_load_raises_on_embedder_mismatch(built_index, field: str, value):
    """The manifest's EmbedderInfo (as actually written) is mutated to disagree
    with what config's embedder.info reports; load() must reject the mismatch."""
    index_dir, config, _ = built_index
    manifest_path = index_dir / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["embedder"][field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError):
        DenseRetriever.load(index_dir, config)


def test_load_raises_filenotfound_when_no_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        DenseRetriever.load(tmp_path, _config())


def test_load_raises_on_reversed_chunk_meta_sidecar(built_index):
    """R1: a chunk_meta.jsonl whose lines were reversed no longer has row_ids in
    ascending 0..N-1 order, even though its length still matches the store --
    load() must reject it rather than silently resolve wrong citations."""
    index_dir, config, _ = built_index
    meta_path = index_dir / "chunk_meta.jsonl"
    lines = meta_path.read_text().splitlines()
    meta_path.write_text("\n".join(reversed(lines)) + "\n")

    with pytest.raises(ValueError, match="row_id"):
        DenseRetriever.load(index_dir, config)


def test_load_raises_on_truncated_chunk_meta_sidecar(built_index):
    """R1: a chunk_meta.jsonl with trailing lines dropped disagrees in length
    with the vector store and manifest -- load() must reject it."""
    index_dir, config, _ = built_index
    meta_path = index_dir / "chunk_meta.jsonl"
    lines = meta_path.read_text().splitlines()
    meta_path.write_text("\n".join(lines[:-3]) + "\n")

    with pytest.raises(ValueError, match="chunk_meta"):
        DenseRetriever.load(index_dir, config)


def test_max_chunks_per_paper_caps_hits_per_paper(tmp_path: Path):
    """One paper owns the top 3 hits by construction (near-duplicate text); with
    max_chunks_per_paper=1, at most one hit per arxiv_id survives, ranks contiguous."""
    chunks_path = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "index"
    config = _config(retrieval_overrides={"max_chunks_per_paper": 1})

    from docsense.ingest.io import get_record_store, write_chunks
    from docsense.ingest.schema import Chunk

    store = get_record_store("jsonl")
    chunks = [
        Chunk(
            chunk_id="2301.00001_chunk_0000",
            arxiv_id="2301.00001",
            chunk_index=0,
            page_start=1,
            page_end=1,
            text="shared topic alpha bravo charlie",
            token_count=5,
            source="text",
        ),
        Chunk(
            chunk_id="2301.00001_chunk_0001",
            arxiv_id="2301.00001",
            chunk_index=1,
            page_start=2,
            page_end=2,
            text="shared topic alpha bravo delta",
            token_count=5,
            source="text",
        ),
        Chunk(
            chunk_id="2301.00001_chunk_0002",
            arxiv_id="2301.00001",
            chunk_index=2,
            page_start=3,
            page_end=3,
            text="shared topic alpha bravo echo",
            token_count=5,
            source="text",
        ),
        Chunk(
            chunk_id="2301.00002_chunk_0000",
            arxiv_id="2301.00002",
            chunk_index=0,
            page_start=1,
            page_end=1,
            text="completely different vocabulary zinnia willow",
            token_count=5,
            source="text",
        ),
    ]
    write_chunks(chunks, chunks_path, store)
    build_index(chunks_path, index_dir, config)

    retriever = DenseRetriever.load(index_dir, config)
    result = retriever.retrieve("shared topic alpha bravo", top_k=3)

    arxiv_ids = [hit.arxiv_id for hit in result.results]
    assert len(arxiv_ids) == len(set(arxiv_ids))
    assert [hit.rank for hit in result.results] == list(range(len(result.results)))


def test_retrieve_batch_latency_ms_is_mean_per_query(built_index, monkeypatch):
    """R4: `latency_ms` is the batch latency divided by the number of queries,
    not the whole-batch latency stamped onto every result."""
    import docsense.retrieve.dense as dense_module

    index_dir, config, chunks = built_index
    retriever = DenseRetriever.load(index_dir, config)

    ticks = iter([100.0, 100.8])  # perf_counter() called once before, once after
    monkeypatch.setattr(dense_module.time, "perf_counter", lambda: next(ticks))

    queries = [chunks[0].text, chunks[1].text]
    results = retriever.retrieve_batch(queries)

    expected_per_query_ms = (100.8 - 100.0) * 1000.0 / len(queries)
    for result in results:
        assert result.latency_ms == pytest.approx(expected_per_query_ms)


def test_retrieve_batch_empty_list_returns_empty_list(built_index):
    """R15: an empty batch must not raise ZeroDivisionError computing mean latency."""
    index_dir, config, _ = built_index
    retriever = DenseRetriever.load(index_dir, config)

    assert retriever.retrieve_batch([]) == []


@pytest.mark.parametrize("index_type", ["numpy_flat_ip", "faiss_flat_ip"])
def test_retrieve_batch_matches_single_retrieve_with_one_encode_call(
    tmp_path: Path, monkeypatch, index_type: str
):
    """R17: round-trips build_index -> DenseRetriever.load -> retrieve_batch over
    both vector-store backends (not just numpy_flat_ip), so a regression in
    faiss's multi-query search() path (R16) is caught by `check`."""
    if index_type == "faiss_flat_ip":
        pytest.importorskip("faiss")

    chunks_path = tmp_path / "chunks.jsonl"
    chunks = write_sample_chunks(chunks_path)
    index_dir = tmp_path / "index"
    config = _config(index_overrides={"type": index_type})
    build_index(chunks_path, index_dir, config)
    retriever = DenseRetriever.load(index_dir, config)

    queries = [chunks[0].text, chunks[3].text, chunks[7].text]
    expected = [retriever.retrieve(q) for q in queries]

    encode_calls = []
    original_encode = HashEmbedder.encode

    def _counting_encode(self, texts, is_query=False):
        encode_calls.append(list(texts))
        return original_encode(self, texts, is_query=is_query)

    monkeypatch.setattr(HashEmbedder, "encode", _counting_encode)

    batched = retriever.retrieve_batch(queries)

    assert len(encode_calls) == 1
    assert encode_calls[0] == queries
    assert len(batched) == len(expected)
    for actual, want in zip(batched, expected, strict=True):
        assert actual.model_dump(exclude={"latency_ms"}) == want.model_dump(exclude={"latency_ms"})
