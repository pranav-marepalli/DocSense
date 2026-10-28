"""Wires Phase 2's `DenseRetriever.retrieve_batch` to relevance.py/metrics.py.

`evaluate` scores a loaded gold set against a loaded retriever; `run_eval` is
the thin driver scripts/eval.py calls, owning path resolution, loading, and
the report write.
"""

import copy
import hashlib
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import sentence_transformers
import torch

from docsense.eval.io import read_gold_set, write_eval_report
from docsense.eval.metrics import mean_or_none, ndcg_at_k, recall_at_k, reciprocal_rank
from docsense.eval.relevance import credit_hits, relevant_spans
from docsense.eval.schema import EvalReport, GoldExample, QueryScore, RetrievedRef
from docsense.ingest.io import get_record_store
from docsense.retrieve.base import Retriever
from docsense.retrieve.dense import DenseRetriever
from docsense.retrieve.schema import RetrievalResult

# src/docsense/eval/harness.py -> repo root (same pattern as docsense.config.CONFIG_DIR).
REPO_ROOT = Path(__file__).resolve().parents[3]

_PROBE_QUERY = "docsense eval harness provenance probe query"
"""A harmless query issued only to harvest retriever provenance (variant_name,
embedder_model, index_fingerprint) when the gold set is empty and no real
query is available to carry that provenance."""


def _faiss_version() -> str:
    """`faiss-cpu`'s installed version without importing faiss."""
    try:
        return version("faiss-cpu")
    except PackageNotFoundError:
        return "not installed"


def _library_versions() -> dict[str, str]:
    """Versions logged into every report, per CLAUDE.md's reproducibility convention."""
    return {
        "sentence_transformers": sentence_transformers.__version__,
        "faiss": _faiss_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def _sha256_file(path: Path) -> str:
    """sha256 of `path`'s bytes; sha256 of empty bytes if `path` does not exist."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 16), b""):
                digest.update(block)
    return digest.hexdigest()


def score_example(
    example: GoldExample,
    result: RetrievalResult,
    k_values: list[int],
    threshold: int,
    include_detail: bool,
) -> QueryScore:
    """Score one query.

    Returns `scored=False` with `metrics={}` when `example.unanswerable` (no
    relevant span exists, so all three metrics are undefined -- Tradeoffs
    #3), and also when `example.unanswerable` is `False` but no span reaches
    `threshold` (`eval.binary_relevance_threshold` set high enough that this
    example's labeled spans all fall below it): the metrics are equally
    undefined in that case, so it is treated the same way rather than
    propagating an empty `ideal_gains` into `metrics.py` and crashing on an
    internal invariant. Either way such queries are counted in the report
    and excluded from every mean.
    """
    if example.unanswerable or not relevant_spans(example, threshold):
        return QueryScore(
            query_id=example.query_id,
            query=example.query,
            query_type=example.query_type,
            difficulty=example.difficulty,
            num_relevant_spans=0,
            unanswerable=example.unanswerable,
            scored=False,
            metrics={},
            first_relevant_rank=None,
            retrieved=[],
        )

    credited = credit_hits(result.results, example.spans, threshold)

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(credited.first_covered_ranks, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(credited.gains, credited.ideal_gains, k)
    full_depth = len(credited.gains)
    metrics["mrr"] = reciprocal_rank(credited.gains, full_depth, threshold=threshold)

    first_relevant_rank = next(
        (rank for rank, gain in enumerate(credited.gains) if gain >= threshold), None
    )

    retrieved: list[RetrievedRef] = []
    if include_detail:
        retrieved = [
            RetrievedRef(
                chunk_id=hit.chunk_id,
                arxiv_id=hit.arxiv_id,
                page_start=hit.page_start,
                page_end=hit.page_end,
                rank=hit.rank,
                score=hit.score,
                credited_relevance=credited.gains[hit.rank],
            )
            for hit in result.results
        ]

    return QueryScore(
        query_id=example.query_id,
        query=example.query,
        query_type=example.query_type,
        difficulty=example.difficulty,
        num_relevant_spans=len(credited.first_covered_ranks),
        unanswerable=False,
        scored=True,
        metrics=metrics,
        first_relevant_rank=first_relevant_rank,
        retrieved=retrieved,
    )


def evaluate(
    retriever: Retriever,
    gold: list[GoldExample],
    config: dict,
    retriever_config_name: str,
) -> EvalReport:
    """Run `retriever.retrieve_batch` over every gold query at `eval.retrieval_top_k`,
    score each, and aggregate.

    `metrics` are means over scored queries only (None when none are scored,
    which is exactly the empty-gold-set case). `metrics_by_slice` repeats the
    aggregation per value of each field in `eval.slice_by`.

    Raises:
        ValueError: if `eval.retrieval_top_k < max(eval.k_values)`.
    """
    eval_cfg = config["eval"]
    k_values = list(eval_cfg["k_values"])
    threshold = eval_cfg["binary_relevance_threshold"]
    retrieval_top_k = eval_cfg["retrieval_top_k"]
    if k_values and retrieval_top_k < max(k_values):
        raise ValueError(
            f"eval.retrieval_top_k ({retrieval_top_k}) must be >= max(eval.k_values) "
            f"({max(k_values)})"
        )
    include_detail = eval_cfg["per_query_detail"]
    slice_fields = eval_cfg["slice_by"]
    gold_path = Path(eval_cfg["gold_path"])

    started_dt = datetime.now(UTC)

    queries = [example.query for example in gold]
    if queries:
        results = retriever.retrieve_batch(queries, top_k=retrieval_top_k)
        provenance = results[0]
    else:
        results = []
        probe = retriever.retrieve_batch([_PROBE_QUERY], top_k=1)
        provenance = probe[0]

    per_query = [
        score_example(example, result, k_values, threshold, include_detail)
        for example, result in zip(gold, results, strict=True)
    ]
    scored = [query_score for query_score in per_query if query_score.scored]
    num_unanswerable = sum(1 for example in gold if example.unanswerable)

    metric_keys = (
        [f"recall@{k}" for k in k_values] + [f"ndcg@{k}" for k in k_values] + ["mrr"]
    )
    metrics: dict[str, float | None] = {
        key: mean_or_none([query_score.metrics[key] for query_score in scored])
        for key in metric_keys
    }

    metrics_by_slice: dict[str, dict[str, float | None]] = {}
    for field in slice_fields:
        observed_values = sorted({getattr(example, field) for example in gold})
        for value in observed_values:
            slice_scored = [
                query_score for query_score in scored if getattr(query_score, field) == value
            ]
            metrics_by_slice[f"{field}={value}"] = {
                key: mean_or_none([query_score.metrics[key] for query_score in slice_scored])
                for key in metric_keys
            }

    mean_latency_ms = mean_or_none([result.latency_ms for result in results])

    finished_dt = datetime.now(UTC)
    run_id = f"{started_dt.strftime('%Y%m%dT%H%M%SZ')}_{provenance.variant_name}"

    return EvalReport(
        run_id=run_id,
        started_at=started_dt.isoformat(),
        finished_at=finished_dt.isoformat(),
        gold_path=str(gold_path),
        gold_sha256=_sha256_file(gold_path),
        num_queries=len(gold),
        num_scored=len(scored),
        num_unanswerable=num_unanswerable,
        retriever=retriever.name,
        retriever_config=retriever_config_name,
        variant_name=provenance.variant_name,
        embedder_model=provenance.embedder_model,
        index_fingerprint=provenance.index_fingerprint,
        k_values=k_values,
        primary_k=eval_cfg["primary_k"],
        binary_relevance_threshold=threshold,
        retrieval_top_k=retrieval_top_k,
        seed=config["seed"],
        library_versions=_library_versions(),
        metrics=metrics,
        metrics_by_slice=metrics_by_slice,
        mean_latency_ms=mean_latency_ms,
        per_query=per_query,
    )


def run_eval(
    eval_config: dict,
    retriever_config: dict,
    retriever_config_name: str,
    index_dir: Path | None = None,
    gold_path: Path | None = None,
) -> EvalReport:
    """Load the gold set, load a DenseRetriever from `index_dir` (default the
    retriever config's index_dir), call `evaluate`, write the report, return it.

    Does not exit the process.
    """
    eval_config = copy.deepcopy(eval_config)

    if gold_path is not None:
        resolved_gold_path = gold_path
    else:
        resolved_gold_path = Path(eval_config["eval"]["gold_path"])
    if not resolved_gold_path.is_absolute():
        resolved_gold_path = REPO_ROOT / resolved_gold_path
    eval_config["eval"]["gold_path"] = str(resolved_gold_path)

    if index_dir is not None:
        resolved_index_dir = index_dir
    else:
        resolved_index_dir = REPO_ROOT / retriever_config["index_dir"]

    store = get_record_store(eval_config["eval"]["record_format"])
    gold = read_gold_set(resolved_gold_path, store)

    retriever = DenseRetriever.load(resolved_index_dir, retriever_config)

    report = evaluate(retriever, gold, eval_config, retriever_config_name)

    runs_dir = Path(eval_config["eval"]["runs_dir"])
    if not runs_dir.is_absolute():
        runs_dir = REPO_ROOT / runs_dir
    write_eval_report(report, runs_dir)

    return report
