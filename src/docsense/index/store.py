"""Vector persistence, mirroring `docsense.ingest.io.RecordStore`.

All vector collections go through the `VectorStore` abstraction so that swapping
exact search for an approximate index later is a new class plus a config value
(`index.type`) rather than a refactor (Tradeoff #2: approximate indexes are
rejected for this corpus size, but the seam is kept).
"""

from pathlib import Path
from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    """Exact top-k inner-product search over a fixed-dim matrix. One file on disk."""

    index_type: str
    dim: int
    filename: str

    def add(self, vectors: np.ndarray) -> None:
        """Append `vectors` (float32 [n, dim]) in order; row_id is append order."""
        ...

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, row_ids), both [n_queries, min(top_k, len(self))], descending score."""
        ...

    def save(self, path: Path) -> None:
        """Write this store to `path / self.filename`, creating `path` if needed."""
        ...

    def __len__(self) -> int:
        """Number of vectors added."""
        ...


def _validate_vectors(vectors: np.ndarray, dim: int) -> None:
    """Shared `add`/`search` input check for both store implementations."""
    if vectors.dtype != np.float32:
        raise ValueError(f"vectors must be float32, got {vectors.dtype}")
    if vectors.ndim != 2 or vectors.shape[1] != dim:
        raise ValueError(f"vectors must have shape (n, {dim}), got {vectors.shape}")


class FaissFlatIPStore:
    """faiss.IndexFlatIP. index_type='faiss_flat_ip', filename='vectors.faiss'.

    `faiss` is imported lazily inside the methods that need it (not at module
    scope), so a faiss-less environment can still import this module and use
    `NumpyFlatIPStore` (Tradeoff #2).
    """

    index_type = "faiss_flat_ip"
    filename = "vectors.faiss"

    def __init__(self, dim: int, num_threads: int = 1) -> None:
        """`num_threads` is `faiss.omp_set_num_threads`'s argument (config key
        `index.num_threads`, default 1). faiss's OpenMP-parallel multi-query
        `search()` deterministically crashes on machines where torch has already
        imported its own bundled OpenMP runtime (a macOS duplicate-libomp
        clash); pinning faiss to a single thread avoids that clash. Raise this
        in a config known not to hit the conflict for more search parallelism.
        """
        import faiss

        faiss.omp_set_num_threads(num_threads)
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        """Append `vectors` (float32 [n, dim])."""
        _validate_vectors(vectors, self.dim)
        self._index.add(vectors)

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, row_ids), clipped to `min(top_k, len(self))`."""
        _validate_vectors(queries, self.dim)
        k = min(top_k, len(self))
        if k == 0:
            n = queries.shape[0]
            return np.empty((n, 0), dtype=np.float32), np.empty((n, 0), dtype=np.int64)
        scores, row_ids = self._index.search(queries, k)
        return scores.astype(np.float32), row_ids.astype(np.int64)

    def save(self, path: Path) -> None:
        """Write to `path / 'vectors.faiss'`."""
        import faiss

        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / self.filename))

    def _load(self, path: Path) -> None:
        """Replace this store's contents with the index at `path`. Called by `load_vector_store`."""
        import faiss

        index = faiss.read_index(str(path))
        if index.d != self.dim:
            raise ValueError(f"loaded index dim {index.d} does not match expected dim {self.dim}")
        self._index = index

    def __len__(self) -> int:
        return self._index.ntotal


class NumpyFlatIPStore:
    """Brute-force `vectors @ q.T` + argpartition.

    index_type='numpy_flat_ip', filename='vectors.npy'. Exact, so it ranks
    identically to FAISS up to float ties; used in tests and anywhere faiss will
    not install.
    """

    index_type = "numpy_flat_ip"
    filename = "vectors.npy"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.empty((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        """Append `vectors` (float32 [n, dim])."""
        _validate_vectors(vectors, self.dim)
        self._vectors = np.vstack([self._vectors, vectors])

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, row_ids), clipped to `min(top_k, len(self))`."""
        _validate_vectors(queries, self.dim)
        n_queries = queries.shape[0]
        k = min(top_k, len(self))
        if k == 0 or n_queries == 0:
            return (
                np.empty((n_queries, 0), dtype=np.float32),
                np.empty((n_queries, 0), dtype=np.int64),
            )

        sims = queries @ self._vectors.T  # [n_queries, len(self)]
        scores = np.empty((n_queries, k), dtype=np.float32)
        row_ids = np.empty((n_queries, k), dtype=np.int64)
        for i in range(n_queries):
            candidates = np.argpartition(-sims[i], kth=k - 1)[:k]
            order = np.argsort(-sims[i, candidates])
            sorted_candidates = candidates[order]
            row_ids[i] = sorted_candidates
            scores[i] = sims[i, sorted_candidates]
        return scores, row_ids

    def save(self, path: Path) -> None:
        """Write to `path / 'vectors.npy'`."""
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / self.filename, self._vectors)

    def _load(self, path: Path) -> None:
        """Replace this store's contents with the array at `path`. Called by `load_vector_store`."""
        vectors = np.load(path)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"loaded vectors dim {vectors.shape[1]} does not match expected dim {self.dim}"
            )
        self._vectors = vectors.astype(np.float32)

    def __len__(self) -> int:
        return self._vectors.shape[0]


def get_vector_store(index_type: str, dim: int, num_threads: int = 1) -> VectorStore:
    """Empty store by type.

    `num_threads` (config key `index.num_threads`) is only meaningful for
    `"faiss_flat_ip"`; see `FaissFlatIPStore.__init__` for why it defaults to 1.

    Raises:
        ValueError: on an unknown index_type.
    """
    if index_type == "faiss_flat_ip":
        return FaissFlatIPStore(dim, num_threads=num_threads)
    if index_type == "numpy_flat_ip":
        return NumpyFlatIPStore(dim)
    raise ValueError(f"Unknown index_type: {index_type!r}")


def load_vector_store(
    index_dir: Path, index_type: str, dim: int, num_threads: int = 1
) -> VectorStore:
    """Load from `index_dir/<filename>`.

    Raises:
        FileNotFoundError: if the file is absent.
        ValueError: on an unknown index_type, or a dim mismatch with the loaded data.
    """
    store = get_vector_store(index_type, dim, num_threads=num_threads)
    path = index_dir / store.filename
    if not path.is_file():
        raise FileNotFoundError(f"No vector store at {path}")
    store._load(path)
    return store
