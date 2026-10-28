"""Pydantic models for the eval harness. No I/O here (the Phase 1/2 pattern).

`GoldExample` is the only label record in DocSense, written only by a human
keystroke through `scripts/label_eval.py` or by hand-editing
`data/eval/gold_set.jsonl`. It is keyed on a page span (`LabeledSpan`) rather
than a `chunk_id`: `chunk_id`s encode the chunking config
(`chunk_size_tokens`), so a chunk_id-keyed gold set would silently decay the
moment Phase 7 ablates chunking -- the one thing that must survive every
variant. A page span is chunking-invariant and is already the project's
citation unit; `source_chunk_id` is retained on `LabeledSpan` as provenance
only and is never scored.
"""

from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

MAX_RELEVANCE = 2
"""The top of the graded relevance scale. Config only owns the binary
threshold (`eval.binary_relevance_threshold`) that collapses this scale for
Recall/MRR; the scale itself is schema-owned."""


class LabeledSpan(BaseModel):
    """One human relevance judgment, keyed on a page span rather than a chunk_id.

    `relevance == 0` is a *recorded* judgment (the human saw the candidate and
    said no) -- it is never a training negative (CLAUDE.md eval leakage guard,
    §5c of docs/design/phase-3.md): relevance-0 spans are attached to eval
    queries, and mining them as Phase 4 hard negatives would train on the eval
    set.
    """

    arxiv_id: str
    page_start: int
    page_end: int
    relevance: int
    note: str | None
    source_chunk_id: str | None
    """Provenance only: the chunk the labeler was shown. Never scored."""

    @model_validator(mode="after")
    def _check_pages_and_relevance(self) -> "LabeledSpan":
        if self.page_start > self.page_end:
            raise ValueError(
                f"page_start ({self.page_start}) must be <= page_end ({self.page_end})"
            )
        if not (0 <= self.relevance <= MAX_RELEVANCE):
            raise ValueError(f"relevance must be in 0..{MAX_RELEVANCE}, got {self.relevance}")
        return self


class GoldExample(BaseModel):
    """One human-labeled query.

    The only label record in DocSense; written only by a human keystroke
    through `scripts/label_eval.py` or by hand-editing the file.
    """

    query_id: str
    query: str
    query_type: Literal[
        "factoid",
        "definition",
        "method",
        "comparison",
        "numeric_result",
        "table",
        "figure",
        "multi_hop",
    ]
    difficulty: Literal["easy", "medium", "hard"]
    unanswerable: bool
    spans: list[LabeledSpan]
    tags: list[str]
    labeled_by: str
    labeled_at: str
    pooled_from: list[str]
    """Retriever names whose candidates the labeler saw; `[]` means fully manual."""
    pool_k: int | None
    """Pool depth the labeler reviewed; makes pooling bias auditable (§5)."""

    @field_validator("query")
    @classmethod
    def _check_query_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank or whitespace-only")
        return value

    @field_validator("labeled_by")
    @classmethod
    def _check_labeled_by_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("labeled_by must not be blank")
        return value

    @model_validator(mode="after")
    def _check_unanswerable_consistency(self) -> "GoldExample":
        has_relevant_span = any(span.relevance >= 1 for span in self.spans)
        if self.unanswerable and has_relevant_span:
            raise ValueError("unanswerable=True must not have any span with relevance >= 1")
        if not self.unanswerable and not has_relevant_span:
            raise ValueError(
                "unanswerable=False requires at least one span with relevance >= 1"
            )
        return self


class CreditedHits(BaseModel):
    """One query's ranked hits resolved against its spans.

    The only bridge between retrieval output and the pure arithmetic in
    `metrics.py`.
    """

    gains: list[int]
    """Per rank: max relevance of spans NEWLY covered there, else 0."""
    first_covered_ranks: list[int | None]
    """Per relevant span (input order): rank that first covered it."""
    ideal_gains: list[int]
    """Relevances of the relevant spans, descending."""


class RetrievedRef(BaseModel):
    """A retrieved hit reduced for the report: ids, pages, score, credit.

    Carries NO chunk text -- corpus text must not land in a committable
    report (licensing, Phase 1 decision 3) and a report holding
    (query, text) pairs is label-shaped (§5).
    """

    chunk_id: str
    arxiv_id: str
    page_start: int
    page_end: int
    rank: int
    score: float
    credited_relevance: int


class QueryScore(BaseModel):
    """One query's scored result."""

    query_id: str
    query: str
    query_type: str
    difficulty: str
    num_relevant_spans: int
    unanswerable: bool
    scored: bool
    metrics: dict[str, float]
    """"recall@5" / "mrr" / "ndcg@5"; {} when not scored."""
    first_relevant_rank: int | None
    """0-based, None if nothing relevant in the retrieved list."""
    retrieved: list[RetrievedRef]
    """[] unless eval.per_query_detail."""


class EvalReport(BaseModel):
    """One eval run. The "log model and data versions with every eval run" record."""

    run_id: str
    started_at: str
    finished_at: str
    gold_path: str
    gold_sha256: str
    """Labels are a versioned input; a number is meaningless without it."""
    num_queries: int
    num_scored: int
    num_unanswerable: int
    retriever: str
    retriever_config: str
    variant_name: str
    embedder_model: str
    index_fingerprint: str
    k_values: list[int]
    primary_k: int
    binary_relevance_threshold: int
    retrieval_top_k: int
    seed: int
    library_versions: dict[str, str]
    metrics: dict[str, float | None]
    """None when num_scored == 0 (never 0.0 -- an unmeasured metric is never a measured zero)."""
    metrics_by_slice: dict[str, dict[str, float | None]]
    """"query_type=table" -> metrics."""
    mean_latency_ms: float | None
    per_query: list[QueryScore]
