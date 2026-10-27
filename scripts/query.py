#!/usr/bin/env python
"""Query a dense index and print ranked, page-cited passages. Retrieval only --
no generation (Phase 2 scope, see docs/design/phase-2.md decision 1).

Usage:
    python scripts/query.py --config baseline --query TEXT [--top-k N] [--index-dir DIR] [--json]
        [--out PATH]
"""

import argparse
import sys
from pathlib import Path

from docsense.config import load_config
from docsense.retrieve.cite import citations_for, format_citation
from docsense.retrieve.dense import DenseRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_under_eval_dir(path: Path) -> bool:
    """True if any path component is literally "eval" -- the leakage guard (§5):
    a RetrievalResult is the shape of a relevance label, so --out must refuse any
    path a labeling tool might read."""
    parts = set(path.parts) | set(path.resolve().parts)
    return "eval" in parts


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline", help="config/<name>.yaml stem")
    parser.add_argument("--query", required=True, help="query text")
    parser.add_argument("--top-k", type=int, default=None, help="override retrieval.top_k")
    parser.add_argument("--index-dir", type=Path, default=None, help="override config index_dir")
    parser.add_argument("--json", action="store_true", help="print the full RetrievalResult JSON")
    parser.add_argument("--out", type=Path, default=None, help="write one RetrievalResult JSON")
    args = parser.parse_args()

    if not args.query.strip():
        print("error: --query must not be empty or whitespace-only", file=sys.stderr)
        return 1

    if args.out is not None and _is_under_eval_dir(args.out):
        print(f"error: --out refuses a path under an eval directory: {args.out}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    index_dir = args.index_dir if args.index_dir is not None else REPO_ROOT / config["index_dir"]

    try:
        retriever = DenseRetriever.load(index_dir, config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = retriever.retrieve(args.query, top_k=args.top_k)

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        for chunk, citation in zip(result.results, citations_for(result), strict=True):
            snippet = " ".join(chunk.text.split())[:200]
            line = f"[{citation.rank}] {format_citation(citation)} (score={citation.score:.4f})"
            print(f"{line} {snippet}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.model_dump_json(indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
