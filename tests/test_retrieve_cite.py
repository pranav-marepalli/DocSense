"""Acceptance criterion 13: citations_for/citation_for fidelity end-to-end from
chunks.jsonl through the index to the citation, and format_citation rendering."""

from pathlib import Path

from docsense.index.build import build_index
from docsense.retrieve.cite import citation_for, citations_for, format_citation
from docsense.retrieve.dense import DenseRetriever
from docsense.retrieve.schema import Citation, RetrievedChunk
from tests.conftest import write_sample_chunks


def _config() -> dict:
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


def test_citation_for_copies_fields_from_retrieved_chunk():
    hit = RetrievedChunk(
        chunk_id="2301.00001_chunk_0000",
        arxiv_id="2301.00001",
        page_start=3,
        page_end=4,
        text="hit text",
        score=0.5,
        rank=2,
    )
    citation = citation_for(hit)
    assert citation.chunk_id == hit.chunk_id
    assert citation.arxiv_id == hit.arxiv_id
    assert citation.page_start == hit.page_start
    assert citation.page_end == hit.page_end
    assert citation.rank == hit.rank
    assert citation.score == hit.score


def test_citations_for_end_to_end_from_chunks_jsonl_through_index(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = write_sample_chunks(chunks_path)
    index_dir = tmp_path / "index"
    config = _config()
    build_index(chunks_path, index_dir, config)

    retriever = DenseRetriever.load(index_dir, config)
    target = chunks[9]
    result = retriever.retrieve(target.text)

    citations = citations_for(result)
    assert [c.rank for c in citations] == [hit.rank for hit in result.results]

    top = citations[0]
    assert top.chunk_id == target.chunk_id
    assert top.arxiv_id == target.arxiv_id
    assert top.page_start == target.page_start
    assert top.page_end == target.page_end

    for citation, hit in zip(citations, result.results, strict=True):
        assert citation.chunk_id == hit.chunk_id
        assert citation.arxiv_id == hit.arxiv_id
        assert citation.page_start == hit.page_start
        assert citation.page_end == hit.page_end


def test_format_citation_collapses_single_page():
    citation = Citation(
        chunk_id="x", arxiv_id="2301.12345", page_start=4, page_end=4, rank=0, score=0.9
    )
    assert format_citation(citation) == "arXiv:2301.12345 p.4"


def test_format_citation_renders_page_range():
    citation = Citation(
        chunk_id="x", arxiv_id="2301.12345", page_start=4, page_end=5, rank=0, score=0.9
    )
    assert format_citation(citation) == "arXiv:2301.12345 p.4-5"
