#!/usr/bin/env python
"""Flag synthetic training examples that overlap with the eval set.

CLAUDE.md hard rule: synthetic training data must never include passages from
the eval set. This script does an n-gram shingle check so that rule is
verified mechanically instead of by eyeballing diffs.

Usage:
    python scripts/check_eval_leakage.py \
        --eval data/eval/gold.jsonl --train data/train/synthetic.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_SHINGLE_SIZE = 13


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def shingles(text: str, n: int = DEFAULT_SHINGLE_SIZE) -> set[tuple[str, ...]]:
    """Word n-grams ("shingles") of `text`. Empty set if text is shorter than n."""
    tokens = _tokenize(text)
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def find_leaked_examples(
    eval_texts: list[str],
    train_texts: list[str],
    n: int = DEFAULT_SHINGLE_SIZE,
) -> list[int]:
    """Return indices into `train_texts` whose text shares an n-gram with any eval text."""
    eval_shingles: set[tuple[str, ...]] = set()
    for text in eval_texts:
        eval_shingles |= shingles(text, n)

    leaked = []
    for i, text in enumerate(train_texts):
        if shingles(text, n) & eval_shingles:
            leaked.append(i)
    return leaked


def _load_jsonl_field(path: Path, field: str) -> list[str]:
    texts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            texts.append(record[field])
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", required=True, type=Path, help="eval set JSONL path")
    parser.add_argument("--train", required=True, type=Path, help="synthetic training JSONL path")
    parser.add_argument("--field", default="text", help="JSON field holding the passage text")
    parser.add_argument("--shingle-size", type=int, default=DEFAULT_SHINGLE_SIZE)
    args = parser.parse_args()

    eval_texts = _load_jsonl_field(args.eval, args.field)
    train_texts = _load_jsonl_field(args.train, args.field)
    leaked = find_leaked_examples(eval_texts, train_texts, args.shingle_size)

    if leaked:
        preview = leaked[:20]
        suffix = "..." if len(leaked) > 20 else ""
        print(
            f"LEAKAGE DETECTED: {len(leaked)} of {len(train_texts)} training examples "
            f"overlap with the eval set (indices: {preview}{suffix})"
        )
        return 1

    print(f"OK: no overlap between {len(eval_texts)} eval and {len(train_texts)} training examples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
