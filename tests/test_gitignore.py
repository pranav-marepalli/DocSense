"""Acceptance criterion 15 (gitignore half, Phase 1) and acceptance criterion 16
(Phase 2): data/raw/, data/parsed/, and data/index/ are ignored, but
data/manifest.jsonl is not."""

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


def test_data_index_is_ignored():
    assert _is_ignored("data/index/baseline/index_manifest.json")
    assert _is_ignored("data/index/baseline/chunk_meta.jsonl")
    assert _is_ignored("data/index/baseline/vectors.npy")


def test_data_eval_runs_and_derived_are_ignored():
    """Acceptance criterion 16 (Phase 3)."""
    assert _is_ignored("data/eval/runs/20240101T000000Z_baseline.json")
    assert _is_ignored("data/eval/derived/eval_texts.jsonl")


def test_data_eval_gold_set_and_readme_are_not_ignored():
    """Acceptance criterion 16 (Phase 3): the gold set and its README are the
    project's irreplaceable human artifacts and must stay tracked."""
    assert not _is_ignored("data/eval/gold_set.jsonl")
    assert not _is_ignored("data/eval/README.md")
