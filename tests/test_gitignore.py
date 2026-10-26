"""Acceptance criterion 15 (gitignore half): data/raw/ and data/parsed/ are
ignored, but data/manifest.jsonl is not."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_ignored(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative_path],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def test_data_raw_and_parsed_are_ignored():
    assert _is_ignored("data/raw/2301.00001v1.pdf")
    assert _is_ignored("data/parsed/chunks.jsonl")
    assert _is_ignored("data/parsed/papers/2301.00001.json")


def test_manifest_is_not_ignored():
    assert not _is_ignored("data/manifest.jsonl")
