"""Acceptance criterion 14: scripts/build_index.py and scripts/query.py run
end-to-end on config/offline_smoke.yaml; query.py's exit codes and --out guard."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.conftest import write_sample_chunks

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    """Load scripts/<name>.py as a standalone module (scripts/ is not a package)."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"docsense_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def smoke_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    chunks_path = tmp_path / "chunks.jsonl"
    write_sample_chunks(chunks_path)
    index_dir = tmp_path / "index"
    build_script = _load_script("build_index")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_index.py",
            "--config",
            "offline_smoke",
            "--chunks",
            str(chunks_path),
            "--index-dir",
            str(index_dir),
        ],
    )
    exit_code = build_script.main()
    assert exit_code == 0
    assert (index_dir / "index_manifest.json").exists()
    return index_dir


def test_build_and_query_scripts_run_end_to_end(
    smoke_setup: Path, monkeypatch: pytest.MonkeyPatch
):
    index_dir = smoke_setup
    query_script = _load_script("query")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "query.py",
            "--config",
            "offline_smoke",
            "--query",
            "alpha bravo charlie delta",
            "--index-dir",
            str(index_dir),
        ],
    )

    exit_code = query_script.main()
    assert exit_code == 0


def test_query_script_prints_one_citation_line_per_hit(
    smoke_setup: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    index_dir = smoke_setup
    query_script = _load_script("query")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "query.py",
            "--config",
            "offline_smoke",
            "--query",
            "alpha bravo charlie delta",
            "--index-dir",
            str(index_dir),
            "--top-k",
            "3",
        ],
    )

    exit_code = query_script.main()
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.startswith("[")]

    assert exit_code == 0
    assert len(lines) == 3
    for line in lines:
        assert "arXiv:" in line
        assert "p." in line


def test_query_script_json_output_is_one_retrieval_result(
    smoke_setup: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    index_dir = smoke_setup
    query_script = _load_script("query")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "query.py",
            "--config",
            "offline_smoke",
            "--query",
            "alpha bravo charlie delta",
            "--index-dir",
            str(index_dir),
            "--json",
        ],
    )

    exit_code = query_script.main()
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["retriever"] == "dense"
    assert "results" in payload


def test_query_script_exits_nonzero_on_empty_query(
    smoke_setup: Path, monkeypatch: pytest.MonkeyPatch
):
    index_dir = smoke_setup
    query_script = _load_script("query")
    monkeypatch.setattr(
        sys,
        "argv",
        ["query.py", "--config", "offline_smoke", "--query", "   ", "--index-dir", str(index_dir)],
    )

    assert query_script.main() != 0


def test_query_script_exits_nonzero_on_missing_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    query_script = _load_script("query")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "query.py",
            "--config",
            "offline_smoke",
            "--query",
            "hello",
            "--index-dir",
            str(tmp_path / "no_such_index"),
        ],
    )

    assert query_script.main() != 0


def test_query_script_out_refuses_eval_directory_path(
    smoke_setup: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    index_dir = smoke_setup
    query_script = _load_script("query")
    out_path = tmp_path / "eval" / "leaked_result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "query.py",
            "--config",
            "offline_smoke",
            "--query",
            "alpha bravo",
            "--index-dir",
            str(index_dir),
            "--out",
            str(out_path),
        ],
    )

    exit_code = query_script.main()
    assert exit_code != 0
    assert not out_path.exists()


def test_query_script_out_writes_result_json_outside_eval(
    smoke_setup: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    index_dir = smoke_setup
    query_script = _load_script("query")
    out_path = tmp_path / "results" / "query_result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "query.py",
            "--config",
            "offline_smoke",
            "--query",
            "alpha bravo",
            "--index-dir",
            str(index_dir),
            "--out",
            str(out_path),
        ],
    )

    exit_code = query_script.main()
    assert exit_code == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["retriever"] == "dense"
