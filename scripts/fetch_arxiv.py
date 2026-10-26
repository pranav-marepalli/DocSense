#!/usr/bin/env python
"""Fetch the arXiv corpus named by `config/ingest.yaml` into `data/raw/`.

Resumable and category-stratified: re-running this script only downloads
papers not already in `data/manifest.jsonl`.

Usage:
    python scripts/fetch_arxiv.py --config ingest [--max-papers N] [--dry-run]
"""

import argparse
import sys
from pathlib import Path

from docsense.config import load_config
from docsense.ingest.fetch_arxiv import (
    fetch_corpus,
    iter_new_candidates,
    iter_rebuild_candidates,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _list_new_candidates(config: dict, manifest_path: Path) -> list[dict[str, str]]:
    """Dry-run: list the *new* papers `fetch_corpus` would download.

    Delegates to `iter_new_candidates`, the same discovery loop `fetch_corpus` uses
    (R4), so a dry-run can never drift from what a real run actually fetches. Driven
    as a plain `for` (no `.send()`), so counting is offer-based -- the right estimate
    here since a dry run never downloads anything and so cannot know what would
    succeed (R20).
    """
    return [
        {"arxiv_id": entry["arxiv_id"], "query_category": category}
        for category, entry in iter_new_candidates(config, manifest_path)
    ]


def _list_rebuild_candidates(config: dict, manifest_path: Path, pdf_dir: Path) -> list[str]:
    """Dry-run: list the manifest records a real run would re-download to rebuild
    the corpus (decision 2's rebuild path -- PDF missing on disk).

    Delegates to `iter_rebuild_candidates`, the same predicate `fetch_corpus`'s
    rebuild pass uses (R19), so a dry-run never misses that this is exactly the case
    where a full manifest with no local PDFs (e.g. a fresh clone) downloads the whole
    corpus.
    """
    return [record.arxiv_id for record in iter_rebuild_candidates(config, manifest_path, pdf_dir)]


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ingest", help="config/<name>.yaml stem")
    parser.add_argument("--max-papers", type=int, default=None, help="override fetch.max_papers")
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be downloaded; write nothing"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_papers is not None:
        config["fetch"]["max_papers"] = args.max_papers

    manifest_path = REPO_ROOT / config["manifest_path"]
    pdf_dir = REPO_ROOT / config["pdf_dir"]

    if args.dry_run:
        rebuild_candidates = _list_rebuild_candidates(config, manifest_path, pdf_dir)
        new_candidates = _list_new_candidates(config, manifest_path)
        print(
            f"Dry run: {len(rebuild_candidates)} manifest paper(s) would be re-downloaded "
            "to rebuild the corpus (PDF missing on disk):"
        )
        for arxiv_id in rebuild_candidates:
            print(f"  {arxiv_id}")
        print(f"Dry run: {len(new_candidates)} new paper(s) would be downloaded:")
        for candidate in new_candidates:
            print(f"  {candidate['arxiv_id']} (category={candidate['query_category']})")
        return 0

    report = fetch_corpus(config, manifest_path, pdf_dir)
    print(f"started_at: {report.started_at}")
    print(f"finished_at: {report.finished_at}")
    print(f"already_in_manifest: {report.already_in_manifest}")
    print(f"newly_fetched: {report.newly_fetched} / max_papers {report.max_papers}")
    print("per-category target vs. actual:")
    for category, target in report.category_targets.items():
        actual = report.newly_fetched_by_category.get(category, 0)
        print(f"  {category}: target={target} newly_fetched={actual}")
    print(f"api_requests: {report.api_requests}")
    if report.failed:
        print(f"failed ({len(report.failed)}):")
        for failure in report.failed:
            print(f"  {failure['arxiv_id']}: {failure['error']}")

    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
