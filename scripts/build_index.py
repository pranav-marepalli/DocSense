#!/usr/bin/env python
"""Embed data/parsed/chunks.jsonl into a dense vector index.

Usage:
    python scripts/build_index.py --config baseline [--chunks PATH] [--index-dir DIR] [--force]
"""

import argparse
import sys
from pathlib import Path

from docsense.config import load_config
from docsense.index.build import build_index

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline", help="config/<name>.yaml stem")
    parser.add_argument("--chunks", type=Path, default=None, help="override config chunks_path")
    parser.add_argument("--index-dir", type=Path, default=None, help="override config index_dir")
    parser.add_argument("--force", action="store_true", help="re-embed even if inputs unchanged")
    args = parser.parse_args()

    config = load_config(args.config)
    chunks_path = args.chunks if args.chunks is not None else REPO_ROOT / config["chunks_path"]
    index_dir = args.index_dir if args.index_dir is not None else REPO_ROOT / config["index_dir"]

    try:
        manifest = build_index(chunks_path, index_dir, config, force=args.force)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"variant_name: {manifest.variant_name}")
    print(f"built_at: {manifest.built_at}")
    print(
        f"embedder: {manifest.embedder.type} ({manifest.embedder.model_name}, "
        f"dim={manifest.embedder.dim})"
    )
    print(f"index_type: {manifest.index_type}")
    print(f"num_vectors: {manifest.num_vectors}")
    print(f"num_chunks: {manifest.num_chunks}")
    print(f"num_papers: {manifest.num_papers}")
    print(f"chunks_sha256: {manifest.chunks_sha256}")
    print(f"fingerprint: {manifest.fingerprint}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
