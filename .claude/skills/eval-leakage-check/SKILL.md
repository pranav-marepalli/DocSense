---
name: eval-leakage-check
description: Verify synthetic training data does not overlap with the eval set. Use this whenever synthetic training data is generated or changed (phase 4+), and as part of reviewer's ML-correctness check — CLAUDE.md's hard rule is that eval passages must never leak into training data.
---

Run `scripts/check_eval_leakage.py` against the eval set and the synthetic
training file being checked:

```
.venv/bin/python scripts/check_eval_leakage.py --eval <eval_path> --train <train_path>
```

Exit code 0 means clean. Exit code 1 means leakage was detected — treat this
as a blocker, not a should-fix: report the overlapping indices and stop
rather than suggesting the data be used anyway.

If the field holding passage text isn't `text`, pass `--field <name>`. Adjust
shingle size with `--shingle-size` only if the default (13-word n-grams)
produces false positives on short passages — do not raise it just to make
a failure go away.
