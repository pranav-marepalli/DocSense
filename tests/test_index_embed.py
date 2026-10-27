"""Acceptance criteria 2 and 3: embedder factory, HashEmbedder determinism, and
SentenceTransformerEmbedder's prefixing with no real model load."""

import numpy as np
import pytest

from docsense.index import embed as embed_module
from docsense.index.embed import (
    HashEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)


def _base_config(embedder_overrides: dict | None = None) -> dict:
    embedder = {
        "type": "hash",
        "model_name": "BAAI/bge-small-en-v1.5",
        "dim": 16,
        "normalize": True,
        "batch_size": 8,
        "device": None,
        "max_seq_length": 512,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "trust_remote_code": False,
    }
    embedder.update(embedder_overrides or {})
    return {"seed": 0, "embedder": embedder}


def test_get_embedder_unknown_type_raises():
    with pytest.raises(ValueError):
        get_embedder(_base_config({"type": "nope"}))


def test_get_embedder_hash_returns_hash_embedder():
    embedder = get_embedder(_base_config())
    assert isinstance(embedder, HashEmbedder)
    assert embedder.info.type == "hash"


def test_hash_embedder_deterministic_across_instances():
    a = HashEmbedder(dim=16, normalize=True, seed=0)
    b = HashEmbedder(dim=16, normalize=True, seed=0)
    vec_a = a.encode(["the quick brown fox"])
    vec_b = b.encode(["the quick brown fox"])
    assert np.array_equal(vec_a, vec_b)


def test_hash_embedder_shape_dtype_and_unit_norm():
    embedder = HashEmbedder(dim=16, normalize=True, seed=0)
    vectors = embedder.encode(["hello world", "another sentence here"])
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 16)
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_hash_embedder_different_texts_different_vectors():
    embedder = HashEmbedder(dim=32, normalize=True, seed=0)
    vectors = embedder.encode(["alpha bravo charlie", "delta echo foxtrot"])
    assert not np.array_equal(vectors[0], vectors[1])


def test_hash_embedder_warns_on_construction():
    with pytest.warns(UserWarning):
        HashEmbedder(dim=8, normalize=True, seed=0)


class _FakeSentenceTransformer:
    """Captures every `encode()` call's inputs; stands in for a real model."""

    instances: list["_FakeSentenceTransformer"] = []

    def __init__(self, model_name: str, device=None, trust_remote_code=False):
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.max_seq_length = None
        self.calls: list[list[str]] = []
        _FakeSentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, inputs, batch_size, convert_to_numpy, normalize_embeddings):
        self.calls.append(list(inputs))
        return np.ones((len(inputs), 4), dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_fake_instances():
    _FakeSentenceTransformer.instances.clear()
    yield
    _FakeSentenceTransformer.instances.clear()


def test_sentence_transformer_embedder_construction_loads_no_model(monkeypatch):
    monkeypatch.setattr(embed_module, "SentenceTransformer", _FakeSentenceTransformer)
    SentenceTransformerEmbedder(
        model_name="BAAI/bge-small-en-v1.5",
        dim=4,
        normalize=True,
        batch_size=8,
        device=None,
        max_seq_length=512,
        query_prefix="query: ",
        passage_prefix="passage: ",
        seed=0,
    )
    assert _FakeSentenceTransformer.instances == []


def test_sentence_transformer_embedder_applies_query_and_passage_prefixes(monkeypatch):
    monkeypatch.setattr(embed_module, "SentenceTransformer", _FakeSentenceTransformer)
    embedder = SentenceTransformerEmbedder(
        model_name="BAAI/bge-small-en-v1.5",
        dim=4,
        normalize=True,
        batch_size=8,
        device=None,
        max_seq_length=512,
        query_prefix="query: ",
        passage_prefix="passage: ",
        seed=0,
    )

    embedder.encode(["hello"], is_query=True)
    embedder.encode(["world"], is_query=False)

    fake_model = _FakeSentenceTransformer.instances[0]
    assert fake_model.calls == [["query: hello"], ["passage: world"]]


def test_get_embedder_sentence_transformer_builds_from_config(monkeypatch):
    monkeypatch.setattr(embed_module, "SentenceTransformer", _FakeSentenceTransformer)
    embedder = get_embedder(
        _base_config(
            {
                "type": "sentence_transformer",
                "dim": 4,
                "query_prefix": "q: ",
                "passage_prefix": "p: ",
            }
        )
    )
    assert isinstance(embedder, SentenceTransformerEmbedder)
    vectors = embedder.encode(["hi"], is_query=True)
    assert vectors.shape == (1, 4)
    assert _FakeSentenceTransformer.instances[0].calls == [["q: hi"]]


def test_info_with_unset_dim_discovers_dim_from_loaded_model(monkeypatch):
    """R3(i): `dim=None` in config -- constructing the embedder loads no model,
    but the first `.info` access triggers real dimension discovery and reports
    the fake model's actual dim (4)."""
    monkeypatch.setattr(embed_module, "SentenceTransformer", _FakeSentenceTransformer)
    embedder = SentenceTransformerEmbedder(
        model_name="BAAI/bge-small-en-v1.5",
        dim=None,
        normalize=True,
        batch_size=8,
        device=None,
        max_seq_length=512,
        query_prefix="query: ",
        passage_prefix="passage: ",
        seed=0,
    )
    assert _FakeSentenceTransformer.instances == []

    info = embedder.info

    assert info.dim == 4
    assert len(_FakeSentenceTransformer.instances) == 1


def test_mismatched_configured_dim_raises_on_encode(monkeypatch):
    """R3(ii): a configured `dim` that disagrees with the real model's output dim
    (4, per `_FakeSentenceTransformer`) raises `ValueError` when the model loads."""
    monkeypatch.setattr(embed_module, "SentenceTransformer", _FakeSentenceTransformer)
    embedder = SentenceTransformerEmbedder(
        model_name="BAAI/bge-small-en-v1.5",
        dim=999,
        normalize=True,
        batch_size=8,
        device=None,
        max_seq_length=512,
        query_prefix="query: ",
        passage_prefix="passage: ",
        seed=0,
    )

    with pytest.raises(ValueError, match="999"):
        embedder.encode(["hello"])


def test_get_embedder_forwards_trust_remote_code_device_and_max_seq_length(monkeypatch):
    """R3(iii): `trust_remote_code`, `device`, and `max_seq_length` from config reach
    the constructed model exactly."""
    monkeypatch.setattr(embed_module, "SentenceTransformer", _FakeSentenceTransformer)
    embedder = get_embedder(
        _base_config(
            {
                "type": "sentence_transformer",
                "dim": 4,
                "device": "mps",
                "max_seq_length": 256,
                "trust_remote_code": True,
            }
        )
    )
    embedder.encode(["hi"], is_query=True)

    fake_model = _FakeSentenceTransformer.instances[0]
    assert fake_model.device == "mps"
    assert fake_model.trust_remote_code is True
    assert fake_model.max_seq_length == 256
