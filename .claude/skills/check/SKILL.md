---
name: check
description: Run the DocSense test + lint gate (pytest -q and ruff check .). Use this instead of re-deriving the commands whenever CLAUDE.md's "before declaring any work done" rule applies — after implementing a phase, after a reviewer fix, or before reporting a task finished.
---

Run, in order, from the project root:

1. `.venv/bin/python -m pytest -q`
2. `.venv/bin/ruff check .`

Report both results together as PASS/FAIL. If either fails, show the failing
test names or lint rule IDs and the file:line, not the full raw output. Do not
report work as done while either command fails.
