#!/usr/bin/env python
"""Export eval-side queries and labeled passages for scripts/check_eval_leakage.py.

Usage:
    python scripts/export_eval_passages.py --config eval [--index-dir DIR] [--out PATH]
"""

import argparse
import sys
from pathlib import Path

from docsense.config import load_config
from docsense.eval.io import export_eval_texts, is_gold_like, read_gold_set
from docsense.index.build import chunk_meta_filename
from docsense.index.schema import IndexedChunk
from docsense.ingest.io import get_record_store

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("data/eval/derived/eval_texts.jsonl")


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="eval", help="config/<name>.yaml stem")
    parser.add_argument("--index-dir", type=Path, default=None, help="override retriever index_dir")
    parser.add_argument("--out", type=Path, default=None, help="output path")
    args = parser.parse_args()

    eval_config = load_config(args.config)
    retriever_config = load_config(eval_config["retriever_config"])

    gold_path = REPO_ROOT / eval_config["eval"]["gold_path"]
    eval_store = get_record_store(eval_config["eval"]["record_format"])

    out_path = args.out if args.out is not None else REPO_ROOT / DEFAULT_OUT
    if is_gold_like(out_path, gold_path):
        print(
            f"error: --out refuses a gold-shaped path: {out_path} "
            "(this would overwrite or be mistaken for the human-labeled gold set)",
            file=sys.stderr,
        )
        return 1

    gold = read_gold_set(gold_path, eval_store)

    index_dir = (
        args.index_dir if args.index_dir is not None else REPO_ROOT / retriever_config["index_dir"]
    )
    record_store = get_record_store(retriever_config["record_format"])
    meta_path = index_dir / chunk_meta_filename(record_store)
    if not meta_path.is_file():
        print(f"error: no chunk metadata at {meta_path}; build the index first", file=sys.stderr)
        return 1
    chunk_meta = [IndexedChunk.model_validate(r) for r in record_store.read_all(meta_path)]

    count = export_eval_texts(gold, chunk_meta, out_path, eval_store, gold_path=gold_path)

    print(f"wrote {count} record(s) to {out_path}")
    print(
        "next: .venv/bin/python scripts/check_eval_leakage.py "
        f"--eval {out_path} --train <synthetic_training_path.jsonl> --field text"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
