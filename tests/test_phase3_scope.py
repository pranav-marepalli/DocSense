"""Acceptance criterion §6.16 (retrieve_batch half) and §6.17 (README content)."""

import inspect
from pathlib import Path

from docsense.retrieve.base import Retriever
from docsense.retrieve.dense import DenseRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_retriever_protocol_declares_retrieve_batch():
    assert "retrieve_batch" in dir(Retriever)
    assert "retrieve_batch" in Retriever.__annotations__ or hasattr(Retriever, "retrieve_batch")


def test_dense_retriever_satisfies_retriever_protocol_signature():
    assert hasattr(DenseRetriever, "retrieve_batch")
    sig = inspect.signature(DenseRetriever.retrieve_batch)
    assert list(sig.parameters)[:3] == ["self", "queries", "top_k"]
    protocol_sig = inspect.signature(Retriever.retrieve_batch)
    assert list(protocol_sig.parameters)[:3] == ["self", "queries", "top_k"]


def test_eval_readme_documents_required_sections():
    text = (REPO_ROOT / "data" / "eval" / "README.md").read_text()
    # §7 human protocol.
    assert "label_eval.py" in text
    assert "suggest-papers" in text
    # File format, field by field.
    assert "query_id" in text
    assert "labeled_by" in text
    assert "pooled_from" in text
    assert "pool_k" in text
    # First-credit nDCG deviation (Tradeoffs #4).
    assert "first-credit" in text.lower()
    # Relevance-0-is-not-a-negative rule (§5c).
    assert "not a training negative" in text.lower() or "not a negative" in text.lower()
