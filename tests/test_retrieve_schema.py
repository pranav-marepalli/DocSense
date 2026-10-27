"""Acceptance criterion 1 (retrieve half): every model in retrieve/schema.py
round-trips model_dump_json() -> model_validate_json() to an equal object."""

from docsense.retrieve.schema import Citation, RetrievalResult, RetrievedChunk


def _roundtrip(model):
    cls = type(model)
    return cls.model_validate_json(model.model_dump_json())


def test_retrieved_chunk_roundtrips():
    model = RetrievedChunk(
        chunk_id="2301.00001_chunk_0000",
        arxiv_id="2301.00001",
        page_start=1,
        page_end=2,
        text="hit text",
        score=0.987,
        rank=0,
    )
    assert _roundtrip(model) == model


def test_citation_roundtrips():
    model = Citation(
        chunk_id="2301.00001_chunk_0000",
        arxiv_id="2301.00001",
        page_start=1,
        page_end=2,
        rank=0,
        score=0.987,
    )
    assert _roundtrip(model) == model


def test_retrieval_result_roundtrips():
    model = RetrievalResult(
        query="what is this paper about",
        retriever="dense",
        top_k=5,
        variant_name="baseline",
        embedder_model="hash",
        index_fingerprint="c" * 64,
        latency_ms=12.3,
        results=[
            RetrievedChunk(
                chunk_id="2301.00001_chunk_0000",
                arxiv_id="2301.00001",
                page_start=1,
                page_end=1,
                text="hit text",
                score=0.9,
                rank=0,
            )
        ],
    )
    assert _roundtrip(model) == model
