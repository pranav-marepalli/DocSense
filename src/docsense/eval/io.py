"""Persistence for the eval harness.

There is deliberately no `write_gold_set` (or equivalent) function anywhere in
`src/docsense/`: the only writer of gold labels is `scripts/label_eval.py`,
which appends one `GoldExample` per completed human session. CLAUDE.md's hard
rule -- "Never create, edit, or fabricate evaluation labels or results" -- is
load-bearing here.
"""

from pathlib import Path

from docsense.eval.schema import EvalReport, GoldExample
from docsense.index.schema import IndexedChunk
from docsense.ingest.io import RecordStore


def _normalize_query(query: str) -> str:
    """Whitespace-normalized, case-folded query text, for duplicate detection."""
    return " ".join(query.split()).casefold()


def read_gold_set(path: Path, store: RecordStore) -> list[GoldExample]:
    """Read and validate every GoldExample from `path`.

    Returns `[]` if the file is absent or empty (the end-of-phase state).

    Raises:
        ValueError: on a duplicate `query_id` or a duplicate
            whitespace-normalized, case-folded query.
        pydantic.ValidationError: on a malformed line.
    """
    examples = [GoldExample.model_validate(record) for record in store.read_all(path)]

    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for example in examples:
        if example.query_id in seen_ids:
            raise ValueError(f"duplicate query_id: {example.query_id!r}")
        seen_ids.add(example.query_id)

        normalized = _normalize_query(example.query)
        if normalized in seen_queries:
            raise ValueError(
                f"duplicate query text (case/whitespace-insensitive): {example.query!r}"
            )
        seen_queries.add(normalized)

    return examples


def is_gold_like(path: Path, gold_path: Path | None) -> bool:
    """True if `path` is the configured gold path or looks like a gold file by name.

    Shared leakage guard used by every writer in this module (and by
    `scripts/export_eval_passages.py`) before it touches a path: an
    `EvalReport` or an exported eval-text record is exactly gold-shaped (a
    query paired with chunk ids/relevance grades, or a (query_id, text)
    pair), so writing one over a gold-shaped path would silently destroy the
    one irreplaceable human artifact in this repo (§5b of
    docs/design/phase-3.md). `gold_path` may be `None` when the caller has no
    specific configured gold path to compare against; the name-based check
    still applies.
    """
    if "gold" in path.name.lower():
        return True
    if gold_path is None:
        return False
    try:
        return path.resolve() == gold_path.resolve()
    except OSError:
        return False


def write_eval_report(report: EvalReport, runs_dir: Path) -> Path:
    """Write `runs_dir/<run_id>.json` (model_dump_json, indent=2).

    Raises:
        ValueError: if the target resolves to the configured gold_path or to
            any `*gold*` filename (§5 leakage guard).
    """
    path = runs_dir / f"{report.run_id}.json"
    gold_path = Path(report.gold_path)
    if is_gold_like(path, gold_path):
        raise ValueError(
            f"refusing to write an EvalReport to a gold-shaped path: {path} "
            "(an EvalReport is label-shaped and must never be mistaken for a gold file)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2))
    return path


def export_eval_texts(
    gold: list[GoldExample],
    chunk_meta: list[IndexedChunk],
    out_path: Path,
    store: RecordStore,
    gold_path: Path | None = None,
) -> int:
    """Write the eval-side input for `eval-leakage-check`.

    One record per gold query (`{"kind": "query", "query_id", "text": query}`)
    and per chunk overlapping any labeled span
    (`{"kind": "passage", "query_id", "chunk_id", "text": chunk.text}`), so
    `check_eval_leakage.py --field text` catches both leaked passages and
    leaked queries. Derived and gitignored; never a label source.

    `gold_path`, when given, is the configured `eval.gold_path` this call was
    read from -- passed through so `is_gold_like` can refuse an `out_path`
    that resolves to it, in addition to the always-on `*gold*` filename check.

    Args:
        gold: The gold examples to export query records for.
        chunk_meta: The index's chunk metadata, to resolve labeled spans to text.
        out_path: Where to write the derived (gitignored) records.
        store: The `RecordStore` backend to write with.
        gold_path: The configured gold path, for the exact-path leakage guard.

    Returns:
        The record count written.

    Raises:
        ValueError: if `out_path` is the configured gold path or looks like a
            gold file by name (§5b leakage guard) -- these derived,
            label-shaped records must never overwrite the human gold set.
    """
    from docsense.eval.relevance import spans_overlap

    if is_gold_like(out_path, gold_path):
        raise ValueError(
            f"refusing to write derived eval texts to a gold-shaped path: {out_path} "
            "(these (query_id, text) records are label-shaped and must never be "
            "mistaken for, or overwrite, the human gold set)"
        )

    records: list[dict[str, str]] = []
    for example in gold:
        records.append({"kind": "query", "query_id": example.query_id, "text": example.query})

    for example in gold:
        seen_chunk_ids: set[str] = set()
        for span in example.spans:
            for chunk in chunk_meta:
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                if chunk.arxiv_id != span.arxiv_id:
                    continue
                if not spans_overlap(
                    chunk.page_start, chunk.page_end, span.page_start, span.page_end
                ):
                    continue
                seen_chunk_ids.add(chunk.chunk_id)
                records.append(
                    {
                        "kind": "passage",
                        "query_id": example.query_id,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                    }
                )

    if out_path.exists():
        out_path.unlink()
    store.append(records, out_path)
    return len(records)
