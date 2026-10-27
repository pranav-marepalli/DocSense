"""Acceptance criterion 4: VectorStore conformance, parameterized over both
implementations."""

from pathlib import Path

import numpy as np
import pytest

from docsense.index.store import (
    FaissFlatIPStore,
    NumpyFlatIPStore,
    get_vector_store,
    load_vector_store,
)

DIM = 8


def _unit_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


@pytest.fixture(params=["faiss_flat_ip", "numpy_flat_ip"])
def index_type(request) -> str:
    if request.param == "faiss_flat_ip":
        pytest.importorskip("faiss")
    return request.param


def test_add_then_search_returns_planted_nearest_first(index_type: str):
    store = get_vector_store(index_type, DIM)
    vectors = _unit_vectors(10, DIM, seed=0)
    store.add(vectors)

    query = vectors[3:4].copy()
    scores, row_ids = store.search(query, top_k=3)

    assert row_ids[0, 0] == 3
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_top_k_above_row_count_returns_exactly_len_store(index_type: str):
    store = get_vector_store(index_type, DIM)
    vectors = _unit_vectors(4, DIM, seed=1)
    store.add(vectors)

    scores, row_ids = store.search(vectors[:1], top_k=100)

    assert scores.shape == (1, 4)
    assert row_ids.shape == (1, 4)
    assert len(store) == 4


def test_scores_non_increasing(index_type: str):
    store = get_vector_store(index_type, DIM)
    vectors = _unit_vectors(20, DIM, seed=2)
    store.add(vectors)

    scores, _ = store.search(vectors[:3], top_k=5)
    for row in scores:
        assert all(row[i] >= row[i + 1] for i in range(len(row) - 1))


def test_save_then_load_reproduces_identical_search(index_type: str, tmp_path: Path):
    store = get_vector_store(index_type, DIM)
    vectors = _unit_vectors(15, DIM, seed=3)
    store.add(vectors)

    queries = vectors[:4]
    expected_scores, expected_row_ids = store.search(queries, top_k=5)

    store.save(tmp_path)
    loaded = load_vector_store(tmp_path, index_type, DIM)
    actual_scores, actual_row_ids = loaded.search(queries, top_k=5)

    np.testing.assert_array_equal(actual_row_ids, expected_row_ids)
    np.testing.assert_allclose(actual_scores, expected_scores, atol=1e-5)


def test_add_wrong_dim_or_non_float32_raises(index_type: str):
    store = get_vector_store(index_type, DIM)

    with pytest.raises(ValueError):
        store.add(np.zeros((2, DIM + 1), dtype=np.float32))

    with pytest.raises(ValueError):
        store.add(np.zeros((2, DIM), dtype=np.float64))


def test_get_vector_store_unknown_type_raises():
    with pytest.raises(ValueError):
        get_vector_store("nope", DIM)


def test_len_equals_vectors_added(index_type: str):
    store = get_vector_store(index_type, DIM)
    assert len(store) == 0
    store.add(_unit_vectors(5, DIM, seed=4))
    assert len(store) == 5
    store.add(_unit_vectors(3, DIM, seed=5))
    assert len(store) == 8


def test_load_vector_store_missing_file_raises(tmp_path: Path, index_type: str):
    with pytest.raises(FileNotFoundError):
        load_vector_store(tmp_path, index_type, DIM)


def test_load_vector_store_dim_mismatch_raises(tmp_path: Path, index_type: str):
    store = get_vector_store(index_type, DIM)
    store.add(_unit_vectors(3, DIM, seed=6))
    store.save(tmp_path)

    with pytest.raises(ValueError):
        load_vector_store(tmp_path, index_type, DIM + 1)


def test_faiss_store_type_and_filename():
    pytest.importorskip("faiss")
    store = FaissFlatIPStore(DIM)
    assert store.index_type == "faiss_flat_ip"
    assert store.filename == "vectors.faiss"


def test_numpy_store_type_and_filename():
    store = NumpyFlatIPStore(DIM)
    assert store.index_type == "numpy_flat_ip"
    assert store.filename == "vectors.npy"


def test_numpy_store_works_in_a_faiss_less_environment(monkeypatch, tmp_path: Path):
    """R2: `docsense.index.store` must not import faiss at module scope, so a
    faiss-less environment can still import this module and use
    `NumpyFlatIPStore` end to end (Tradeoff #2). Simulated by making `import
    faiss` raise `ImportError`, as if the package were not installed."""
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("simulated: faiss is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    import importlib

    import docsense.index.store as store_module

    reloaded = importlib.reload(store_module)

    store = reloaded.get_vector_store("numpy_flat_ip", DIM)
    vectors = _unit_vectors(5, DIM, seed=7)
    store.add(vectors)
    scores, row_ids = store.search(vectors[:1], top_k=3)
    assert row_ids[0, 0] == 0

    store.save(tmp_path)
    loaded = reloaded.load_vector_store(tmp_path, "numpy_flat_ip", DIM)
    assert len(loaded) == 5

    with pytest.raises(ImportError):
        reloaded.get_vector_store("faiss_flat_ip", DIM)

    importlib.reload(store_module)
