"""Text -> vector embedding backends.

`Embedder` is the only interface callers may use; `get_embedder` is the config
factory over it. `SentenceTransformerEmbedder` is the real backend (bge/e5-style,
asymmetric query/passage prefixes applied here rather than hardcoded, since a
different model needs different prefixes -- config, not code). `HashEmbedder` is
a deterministic, semantics-free stand-in used only in dev/tests so the whole
`check` suite stays offline.
"""

import warnings
from hashlib import sha256
from typing import Protocol

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from docsense.index.schema import EmbedderInfo


class Embedder(Protocol):
    """Text -> float32 [n, dim]. The only embedding interface callers may use."""

    info: EmbedderInfo

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Encode `texts` as queries (`is_query=True`) or passages, float32 [n, dim]."""
        ...


class SentenceTransformerEmbedder:
    """sentence-transformers backend.

    Applies `query_prefix` when `is_query` else `passage_prefix`. float32,
    L2-normalized when `normalize`, so inner product == cosine. Loads the model
    lazily on first encode (or first `.info` access if `dim` is unset), so
    constructing one downloads nothing.
    """

    def __init__(
        self,
        model_name: str,
        dim: int | None,
        normalize: bool,
        batch_size: int,
        device: str | None,
        max_seq_length: int | None,
        query_prefix: str,
        passage_prefix: str,
        seed: int,
        trust_remote_code: bool = False,
    ) -> None:
        self._model_name = model_name
        self._configured_dim = dim
        self._normalize = normalize
        self._batch_size = batch_size
        self._device = device
        self._max_seq_length = max_seq_length
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._seed = seed
        self._trust_remote_code = trust_remote_code
        self._model: SentenceTransformer | None = None

    @property
    def info(self) -> EmbedderInfo:
        """EmbedderInfo for this embedder; loads the model only if `dim` was unset."""
        dim = self._configured_dim if self._configured_dim is not None else self._model_dim()
        return EmbedderInfo(
            type="sentence_transformer",
            model_name=self._model_name,
            dim=dim,
            normalize=self._normalize,
            max_seq_length=self._max_seq_length,
            query_prefix=self._query_prefix,
            passage_prefix=self._passage_prefix,
        )

    def _model_dim(self) -> int:
        return self._load_model().get_sentence_embedding_dimension()

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            torch.manual_seed(self._seed)
            np.random.seed(self._seed)
            model = SentenceTransformer(
                self._model_name,
                device=self._device,
                trust_remote_code=self._trust_remote_code,
            )
            if self._max_seq_length is not None:
                model.max_seq_length = self._max_seq_length
            if self._configured_dim is not None:
                actual_dim = model.get_sentence_embedding_dimension()
                if actual_dim != self._configured_dim:
                    raise ValueError(
                        f"configured embedder.dim {self._configured_dim} does not match "
                        f"{self._model_name}'s actual dim {actual_dim}"
                    )
            self._model = model
        return self._model

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Encode `texts`, prefixed per `is_query`, as float32 [n, dim]."""
        model = self._load_model()
        prefix = self._query_prefix if is_query else self._passage_prefix
        prefixed = [prefix + text for text in texts]
        vectors = model.encode(
            prefixed,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashEmbedder:
    """Deterministic stub: hashed bag-of-words projected to `dim`, L2-normalized.

    DEV AND TEST ONLY -- no semantics. Warns on construction and records
    type="hash" in EmbedderInfo, so any index built with it is visibly non-real.
    """

    def __init__(self, dim: int, normalize: bool, seed: int) -> None:
        warnings.warn(
            "HashEmbedder has no semantics; it is a deterministic stand-in for dev/tests only.",
            stacklevel=2,
        )
        self._dim = dim
        self._normalize = normalize
        self._seed = seed
        self.info = EmbedderInfo(
            type="hash",
            model_name="hash",
            dim=dim,
            normalize=normalize,
            max_seq_length=None,
            query_prefix="",
            passage_prefix="",
        )

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Hash each token of each text into one of `dim` signed buckets, then normalize."""
        vectors = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = int(sha256(f"{self._seed}:{token}".encode()).hexdigest(), 16)
                column = digest % self._dim
                sign = 1.0 if (digest // self._dim) % 2 == 0 else -1.0
                vectors[row, column] += sign
        if self._normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            vectors = vectors / norms
        return vectors.astype(np.float32)


def get_embedder(config: dict) -> Embedder:
    """Factory over `config["embedder"]["type"]`.

    Raises:
        ValueError: on an unknown embedder type.
    """
    embedder_config = config["embedder"]
    embedder_type = embedder_config["type"]

    if embedder_type == "sentence_transformer":
        return SentenceTransformerEmbedder(
            model_name=embedder_config["model_name"],
            dim=embedder_config["dim"],
            normalize=embedder_config["normalize"],
            batch_size=embedder_config["batch_size"],
            device=embedder_config["device"],
            max_seq_length=embedder_config["max_seq_length"],
            query_prefix=embedder_config["query_prefix"],
            passage_prefix=embedder_config["passage_prefix"],
            seed=config["seed"],
            trust_remote_code=embedder_config["trust_remote_code"],
        )
    if embedder_type == "hash":
        return HashEmbedder(
            dim=embedder_config["dim"],
            normalize=embedder_config["normalize"],
            seed=config["seed"],
        )
    raise ValueError(f"Unknown embedder.type: {embedder_type!r}")
