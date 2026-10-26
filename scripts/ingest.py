#!/usr/bin/env python
"""Extract the fetched arXiv corpus into page-aware chunks, tables, and figures.

Usage:
    python scripts/ingest.py --config ingest [--manifest PATH | --pdf-dir DIR] [--out-dir DIR]
"""

import argparse
import sys
from pathlib import Path

from docsense.config import load_config
from docsense.ingest.pipeline import ingest_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ingest", help="config/<name>.yaml stem")
    parser.add_argument("--manifest", type=Path, default=None, help="manifest.jsonl path")
    parser.add_argument("--pdf-dir", type=Path, default=None, help="directory of PDFs")
    parser.add_argument("--out-dir", type=Path, default=None, help="extraction output root")
    args = parser.parse_args()

    config = load_config(args.config)

    manifest_path = args.manifest
    pdf_dir = args.pdf_dir
    if manifest_path is None and pdf_dir is None:
        manifest_path = REPO_ROOT / config["manifest_path"]
    out_dir = args.out_dir if args.out_dir is not None else REPO_ROOT / config["out_dir"]

    report = ingest_corpus(out_dir, config, manifest_path=manifest_path, pdf_dir=pdf_dir)

    print(f"started_at: {report.started_at}")
    print(f"finished_at: {report.finished_at}")
    print(f"papers_total: {report.papers_total}")
    print(f"papers_ok: {report.papers_ok}")
    print(f"papers_failed: {report.papers_failed}")
    print(f"total_pages: {report.total_pages}")
    print(f"total_chunks: {report.total_chunks}")
    print(f"tables_total: {report.tables_total} ({report.tables_by_method})")
    print(f"figures_total: {report.figures_total}")
    print(f"low_text_layer_pages: {report.low_text_layer_pages}")

    if report.papers_failed:
        print("failed papers:")
        for status in report.papers:
            if status.status == "error":
                print(f"  {status.arxiv_id}: {status.error}")

    return 1 if report.papers_failed else 0


if __name__ == "__main__":
    sys.exit(main())
