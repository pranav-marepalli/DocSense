"""The retriever seam.

`Retriever` is the protocol Phase 4's hybrid/reranked retrievers and a later
generation phase both plug into; `dense.DenseRetriever` is this phase's only
implementation.
"""

from typing import Protocol

from docsense.retrieve.schema import RetrievalResult


class Retriever(Protocol):
    """Query -> RetrievalResult."""

    name: str

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Ranked hits for `query`, `top_k` defaulting to the retriever's configured value."""
        ...

    def retrieve_batch(
        self, queries: list[str], top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Ranked hits for every query in `queries`, same per-query semantics as `retrieve`.

        What Phase 3's eval harness uses.
        """
        ...
