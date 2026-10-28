#!/usr/bin/env python
"""Pool-and-judge human labeling scaffold for the Phase 3 gold set.

This is the ONLY writer of gold labels anywhere in DocSense. Every
`GoldExample` it appends reflects one completed per-candidate human decision
-- there is no default grade, no bulk "accept the pool" path, and no
inference from rank or score. CLAUDE.md's hard rule ("Never create, edit, or
fabricate evaluation labels or results. The human labels the gold set.") is
load-bearing here; see docs/design/phase-3.md §2/§5/§7 for the full design.

Usage:
    python scripts/label_eval.py --config eval --labeled-by NAME
        [--retriever-config NAME] [--pool-k N] [--query-id-prefix q] [--suggest-papers N]

Modes:
    --suggest-papers N   Print N `data/manifest.jsonl` rows (arxiv_id,
                          query_category, title), seeded by `seed`, for the
                          human to skim and write queries about. Emits no
                          labels; writes nothing.
    (default)             Read `labeling.queries_path` (one query per line,
                          `#` comments), skip queries already in the gold
                          set, and for each remaining query retrieve
                          `labeling.pool_k` candidates and prompt a human
                          judgment for each.
"""

import argparse
import random
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

from docsense.config import load_config
from docsense.eval.io import read_gold_set
from docsense.eval.schema import GoldExample, LabeledSpan
from docsense.ingest.io import RecordStore, get_record_store, read_manifest
from docsense.retrieve.base import Retriever
from docsense.retrieve.cite import citation_for, format_citation
from docsense.retrieve.dense import DenseRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]
README_POINTER = "see data/eval/README.md for the labeling workflow"

_QUERY_TYPES: tuple[str, ...] = get_args(GoldExample.model_fields["query_type"].annotation)
_DIFFICULTIES: tuple[str, ...] = get_args(GoldExample.model_fields["difficulty"].annotation)

# "a 2301.00001:4-6 2" -> arxiv_id, page_start, page_end, relevance.
_MANUAL_SPAN_RE = re.compile(r"^a\s+(\S+):(\d+)-(\d+)\s+([0-2])$")


class _QuitSession(Exception):
    """Raised on `q`: end the session now. Already-finalized queries stay appended."""


class _SessionExhausted(Exception):
    """Raised when `prompt_fn` has no more scripted input mid-query.

    Handled identically to `q`: the in-progress query is discarded, everything
    already finalized in this session is already on disk.
    """


def _ask(prompt_fn: Callable[[str], str], text: str) -> str:
    """Call `prompt_fn`, converting input exhaustion into `_SessionExhausted`."""
    try:
        return prompt_fn(text)
    except (EOFError, StopIteration) as exc:
        raise _SessionExhausted from exc


def _normalize_query(query: str) -> str:
    """Whitespace-normalized, case-folded query text (matches `eval.io`'s duplicate check)."""
    return " ".join(query.split()).casefold()


def _read_queries(path: Path) -> list[str]:
    """One query per line; blank lines and `#`-prefixed comments are skipped.

    Raises:
        FileNotFoundError: if `path` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(str(path))
    queries = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        queries.append(stripped)
    return queries


def _parse_manual_span(raw: str) -> LabeledSpan | None:
    """Parse `"a 2301.00001:4-6 2"` into a manually-added span, or `None` if malformed.

    `source_chunk_id` is `None`: a manual span was never shown as a pool
    candidate, so it has no chunk provenance to record.
    """
    match = _MANUAL_SPAN_RE.match(raw.strip())
    if match is None:
        return None
    arxiv_id, page_start, page_end, relevance = match.groups()
    return LabeledSpan(
        arxiv_id=arxiv_id,
        page_start=int(page_start),
        page_end=int(page_end),
        relevance=int(relevance),
        note=None,
        source_chunk_id=None,
    )


def _judge_candidates(
    pool: list,
    snippet_chars: int,
    prompt_fn: Callable[[str], str],
) -> tuple[list[LabeledSpan], bool]:
    """Walk one query's pool, one human decision per candidate.

    Returns `(spans, unanswerable)`. Blank or unrecognized input re-prompts
    the same candidate -- there is no default grade. `a <span>` records a
    manually-added span without advancing past the current candidate, so a
    human can both rescue a missed passage and still grade the candidate in
    front of them. `q` and input exhaustion propagate up (`_QuitSession` /
    `_SessionExhausted`) for the caller to handle.
    """
    spans: list[LabeledSpan] = []
    unanswerable = False
    idx = 0
    while idx < len(pool):
        hit = pool[idx]
        snippet = " ".join(hit.text.split())[:snippet_chars]
        citation_line = format_citation(citation_for(hit))
        prompt_text = (
            f"[{idx + 1}/{len(pool)}] {citation_line}\n{snippet}\n"
            "grade 0/1/2, s=skip, a=add span, u=unanswerable, n=next query, q=quit > "
        )
        raw = _ask(prompt_fn, prompt_text)
        cmd = raw.strip()
        if cmd in ("0", "1", "2"):
            spans.append(
                LabeledSpan(
                    arxiv_id=hit.arxiv_id,
                    page_start=hit.page_start,
                    page_end=hit.page_end,
                    relevance=int(cmd),
                    note=None,
                    source_chunk_id=hit.chunk_id,
                )
            )
            idx += 1
        elif cmd == "s":
            idx += 1
        elif cmd.startswith("a "):
            manual = _parse_manual_span(cmd)
            if manual is not None:
                spans.append(manual)
            # Malformed "a ..." re-prompts the same candidate, same as any
            # other unrecognized input.
        elif cmd == "u":
            unanswerable = True
            break
        elif cmd == "n":
            break
        elif cmd == "q":
            raise _QuitSession
        # else: blank or unrecognized -- re-prompt the same candidate.
    return spans, unanswerable


def _prompt_choice(prompt_fn: Callable[[str], str], text: str, choices: tuple[str, ...]) -> str:
    """Re-prompt until the human answers with one of `choices`."""
    while True:
        raw = _ask(prompt_fn, text)
        cmd = raw.strip()
        if cmd in choices:
            return cmd


def _next_query_id(prefix: str, ordinal: int) -> str:
    """`"q007"`-style id: `prefix` plus a 3-digit, 1-based ordinal."""
    return f"{prefix}{ordinal:03d}"


def _next_ordinal(existing: list[GoldExample], prefix: str) -> int:
    """One past the largest numeric suffix among existing ids sharing `prefix`.

    Using `len(existing) + 1` assumes gold `query_id`s are a dense `1..N`
    sequence, which the gold set's append-only, hand-editable file format
    (re-labeling is "a deliberate, visible file edit", §2) does not
    guarantee: a deleted or renumbered line, a second labeler, or a changed
    `--query-id-prefix` can all leave gaps or a different id scheme. Deriving
    the next ordinal from the max matching suffix instead means a gap (e.g.
    `q001, q002, q004`) advances to `q005`, not a colliding `q003`/`q004`.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_ordinal = 0
    for example in existing:
        match = pattern.match(example.query_id)
        if match is not None:
            max_ordinal = max(max_ordinal, int(match.group(1)))
    return max_ordinal + 1


def run_labeling_session(
    retriever: Retriever,
    queries: list[str],
    gold_path: Path,
    store: RecordStore,
    labeled_by: str,
    pool_k: int,
    snippet_chars: int,
    query_id_prefix: str = "q",
    prompt_fn: Callable[[str], str] = input,
) -> int:
    """Judge each query in `queries` not already present in `gold_path`.

    Appends one `GoldExample` per finalized query, immediately after it is
    finalized -- so a session interrupted partway through still leaves every
    completed judgment on disk. Append-only: never rewrites or deletes an
    existing line, so re-labeling an example later is a deliberate, separate
    file edit.

    A query is skipped when its whitespace-normalized, case-folded text
    already appears in the gold set. `q`, or `prompt_fn` running out of
    scripted input mid-query, ends the session immediately: the in-progress
    query is discarded (never appended), but every query already finalized
    earlier in this session is already written.

    Returns:
        The number of `GoldExample`s appended in this call.
    """
    existing = read_gold_set(gold_path, store)
    seen_queries = {_normalize_query(example.query) for example in existing}
    existing_ids = {example.query_id for example in existing}
    next_ordinal = _next_ordinal(existing, query_id_prefix)

    appended = 0
    for query in queries:
        if _normalize_query(query) in seen_queries:
            continue

        result = retriever.retrieve(query, top_k=pool_k)

        try:
            spans, unanswerable = _judge_candidates(result.results, snippet_chars, prompt_fn)
        except (_QuitSession, _SessionExhausted):
            break

        if unanswerable:
            # Defensive: unanswerable=True must not carry a relevance>=1 span
            # even if one was graded before `u` was entered.
            spans = [span for span in spans if span.relevance < 1]
        elif not any(span.relevance >= 1 for span in spans):
            # Nothing judged relevant and not marked unanswerable: there is
            # nothing to finalize (the schema would reject it). Tell the
            # human explicitly -- their judgments were real decisions, not
            # silently discarded -- and point at "u" in case this query
            # really has no answer in the corpus (R5).
            print(
                f"skipping {query!r}: every graded candidate was 0 (not relevant) and "
                "the query was not marked unanswerable, so there is nothing to record. "
                "If nothing in the corpus answers this query, rerun and press 'u' to "
                "mark it unanswerable instead.",
                file=sys.stderr,
            )
            continue

        try:
            query_type = _prompt_choice(prompt_fn, f"query_type {_QUERY_TYPES} > ", _QUERY_TYPES)
            difficulty = _prompt_choice(
                prompt_fn, f"difficulty {_DIFFICULTIES} > ", _DIFFICULTIES
            )
        except (_QuitSession, _SessionExhausted):
            break

        query_id = _next_query_id(query_id_prefix, next_ordinal)
        while query_id in existing_ids:
            # A collision means the ordinal scheme and the existing ids
            # disagree (e.g. ids written under a different
            # --query-id-prefix). Advance until a free id is found rather
            # than ever writing a duplicate query_id (R2).
            next_ordinal += 1
            query_id = _next_query_id(query_id_prefix, next_ordinal)

        example = GoldExample(
            query_id=query_id,
            query=query,
            query_type=query_type,
            difficulty=difficulty,
            unanswerable=unanswerable,
            spans=spans,
            tags=[],
            labeled_by=labeled_by,
            labeled_at=datetime.now(UTC).isoformat(),
            pooled_from=[retriever.name],
            pool_k=pool_k,
        )
        store.append([example.model_dump(mode="json")], gold_path)
        seen_queries.add(_normalize_query(query))
        existing_ids.add(query_id)
        next_ordinal += 1
        appended += 1

    return appended


def _print_suggested_papers(manifest_path: Path, n: int, seed: int) -> None:
    """Print a seeded sample of `n` manifest rows (arxiv_id, query_category, title).

    Writes nothing -- this emits no labels, only reading material for the
    human to write queries from.
    """
    store = get_record_store("jsonl")
    records = read_manifest(manifest_path, store)
    rng = random.Random(seed)
    sample = records if len(records) <= n else rng.sample(records, n)
    for record in sample:
        print(f"{record.arxiv_id}\t{record.query_category}\t{record.title}")


def main(prompt_fn: Callable[[str], str] = input) -> int:
    """CLI entry point. Returns a process exit code.

    `prompt_fn` is the sole injection point for driving the session
    programmatically (tests); it is never exposed as a CLI flag, so no
    non-interactive path exists in the shipped script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="eval", help="config/<name>.yaml stem")
    parser.add_argument("--labeled-by", default=None, help="who is making these judgments")
    parser.add_argument(
        "--retriever-config", default=None, help="override eval config's retriever_config"
    )
    parser.add_argument("--pool-k", type=int, default=None, help="override labeling.pool_k")
    parser.add_argument("--query-id-prefix", default="q", help="prefix for generated query_ids")
    parser.add_argument(
        "--suggest-papers",
        type=int,
        default=None,
        help="print N data/manifest.jsonl rows and exit; writes nothing",
    )
    args = parser.parse_args()

    eval_config = load_config(args.config)

    if args.suggest_papers is not None:
        ingest_config = load_config("ingest")
        manifest_path = REPO_ROOT / ingest_config["manifest_path"]
        _print_suggested_papers(manifest_path, args.suggest_papers, eval_config["seed"])
        return 0

    if not args.labeled_by or not args.labeled_by.strip():
        print(f"error: --labeled-by is required ({README_POINTER})", file=sys.stderr)
        return 1

    queries_path = REPO_ROOT / eval_config["labeling"]["queries_path"]
    try:
        queries = _read_queries(queries_path)
    except FileNotFoundError:
        print(f"error: no queries file at {queries_path} ({README_POINTER})", file=sys.stderr)
        return 1

    if prompt_fn is input and not sys.stdin.isatty():
        # `prompt_fn is input` means this is the real, shipped, non-test
        # path. stdin reads identically from a pipe or a terminal, so
        # `yes 2 | python scripts/label_eval.py ...` would otherwise be a
        # de facto bulk-label path one shell command away -- undermining the
        # "always requires a human" design intent (R6). Test code always
        # injects its own `prompt_fn`, so this never fires in tests.
        print(
            f"error: label_eval.py requires an interactive terminal; refusing to run "
            f"with non-interactive stdin ({README_POINTER})",
            file=sys.stderr,
        )
        return 1

    retriever_config_name = args.retriever_config or eval_config["retriever_config"]
    retriever_config = load_config(retriever_config_name)
    index_dir = REPO_ROOT / retriever_config["index_dir"]

    try:
        retriever = DenseRetriever.load(index_dir, retriever_config)
    except FileNotFoundError as exc:
        print(f"error: {exc} ({README_POINTER})", file=sys.stderr)
        return 1

    gold_path = REPO_ROOT / eval_config["eval"]["gold_path"]
    store = get_record_store(eval_config["eval"]["record_format"])
    pool_k = args.pool_k if args.pool_k is not None else eval_config["labeling"]["pool_k"]
    snippet_chars = eval_config["labeling"]["snippet_chars"]

    appended = run_labeling_session(
        retriever=retriever,
        queries=queries,
        gold_path=gold_path,
        store=store,
        labeled_by=args.labeled_by,
        pool_k=pool_k,
        snippet_chars=snippet_chars,
        query_id_prefix=args.query_id_prefix,
        prompt_fn=prompt_fn,
    )
    print(f"appended {appended} example(s) to {gold_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
