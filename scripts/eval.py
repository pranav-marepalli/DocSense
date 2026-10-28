#!/usr/bin/env python
"""Run the Phase 3 eval harness over a gold set and a retrieval variant.

Usage:
    python scripts/eval.py --config eval [--retriever-config NAME] [--index-dir DIR]
        [--gold PATH] [--out PATH] [--json] [--allow-empty]
"""

import argparse
import sys
from pathlib import Path

from docsense.config import load_config
from docsense.eval.harness import run_eval

REPO_ROOT = Path(__file__).resolve().parents[1]


def _print_report(report) -> None:  # noqa: ANN001
    """Print one row per metric plus the slice table."""
    print(f"run_id: {report.run_id}")
    print(f"retriever_config: {report.retriever_config}  variant_name: {report.variant_name}")
    print(
        f"num_queries: {report.num_queries}  num_scored: {report.num_scored}  "
        f"num_unanswerable: {report.num_unanswerable}"
    )
    print("metrics:")
    for key, value in report.metrics.items():
        print(f"  {key}: {value if value is not None else 'n/a'}")
    print("metrics_by_slice:")
    for slice_key, slice_metrics in report.metrics_by_slice.items():
        print(f"  {slice_key}:")
        for key, value in slice_metrics.items():
            print(f"    {key}: {value if value is not None else 'n/a'}")


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="eval", help="config/<name>.yaml stem")
    parser.add_argument(
        "--retriever-config", default=None, help="override eval config's retriever_config"
    )
    parser.add_argument("--index-dir", type=Path, default=None, help="override retriever index_dir")
    parser.add_argument("--gold", type=Path, default=None, help="override eval.gold_path")
    parser.add_argument("--out", type=Path, default=None, help="also write a copy of the report")
    parser.add_argument("--json", action="store_true", help="print the full EvalReport JSON")
    parser.add_argument(
        "--allow-empty", action="store_true", help="exit 0 even with zero scorable queries"
    )
    args = parser.parse_args()

    eval_config = load_config(args.config)

    if args.gold is not None and not args.gold.is_file():
        print(f"error: --gold path does not exist: {args.gold}", file=sys.stderr)
        return 1

    retriever_config_name = args.retriever_config or eval_config["retriever_config"]
    retriever_config = load_config(retriever_config_name)

    try:
        report = run_eval(
            eval_config,
            retriever_config,
            retriever_config_name,
            index_dir=args.index_dir,
            gold_path=args.gold,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_report(report)

    if args.out is not None:
        if "gold" in args.out.name.lower():
            print(f"error: --out refuses a gold-shaped filename: {args.out}", file=sys.stderr)
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report.model_dump_json(indent=2))

    min_queries_warn = eval_config["eval"]["min_queries_warn"]
    if report.num_scored < min_queries_warn:
        print(
            f"warning: only {report.num_scored} scored queries "
            f"(< eval.min_queries_warn={min_queries_warn})",
            file=sys.stderr,
        )

    if report.num_scored == 0 and not args.allow_empty:
        print(
            "error: gold set has zero scorable queries; pass --allow-empty to proceed anyway",
            file=sys.stderr,
        )
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
