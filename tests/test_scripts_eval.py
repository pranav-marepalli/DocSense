"""Acceptance criterion §6.13: scripts/eval.py end-to-end against a tmp_path
fixture corpus with --retriever-config offline_smoke."""

import json
import sys
from pathlib import Path

import pytest

from docsense.eval.schema import EvalReport, GoldExample, LabeledSpan
from docsense.index.build import build_index
from scripts import eval as eval_script
from tests.conftest import write_sample_chunks


@pytest.fixture
def built_index(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = write_sample_chunks(chunks_path)
    index_dir = tmp_path / "index"
    config = {
        "variant_name": "offline_smoke",
        "seed": 0,
        "record_format": "jsonl",
        "embedder": {
            "type": "hash",
            "model_name": "BAAI/bge-small-en-v1.5",
            "dim": 384,
            "normalize": True,
            "batch_size": 4,
            "device": None,
            "max_seq_length": 512,
            "query_prefix": "Represent this sentence for searching relevant passages: ",
            "passage_prefix": "",
            "trust_remote_code": False,
        },
        "index": {"type": "numpy_flat_ip", "num_threads": 1},
        "retrieval": {"top_k": 10, "max_chunks_per_paper": None},
    }
    build_index(chunks_path, index_dir, config)
    return index_dir, chunks


def _gold_line(chunk) -> str:
    example = GoldExample(
        query_id="q001",
        query=chunk.text,
        query_type="factoid",
        difficulty="easy",
        unanswerable=False,
        spans=[
            LabeledSpan(
                arxiv_id=chunk.arxiv_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                relevance=2,
                note=None,
                source_chunk_id=chunk.chunk_id,
            )
        ],
        tags=[],
        labeled_by="tester",
        labeled_at="2024-01-01T00:00:00+00:00",
        pooled_from=["dense"],
        pool_k=20,
    )
    return example.model_dump_json()


def _argv(index_dir: Path, gold_path: Path, runs_dir: Path, extra: list[str] | None = None):
    argv = [
        "eval.py",
        "--config",
        "eval",
        "--retriever-config",
        "offline_smoke",
        "--index-dir",
        str(index_dir),
        "--gold",
        str(gold_path),
    ]
    return argv + (extra or [])


def test_exits_3_on_empty_gold_set(built_index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    index_dir, _ = built_index
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text("")
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(sys, "argv", _argv(index_dir, gold_path, runs_dir))
    monkeypatch.setattr(eval_script, "load_config", _patched_load_config(runs_dir))

    assert eval_script.main() == 3


def test_exits_0_with_allow_empty(built_index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    index_dir, _ = built_index
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text("")
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(sys, "argv", _argv(index_dir, gold_path, runs_dir, ["--allow-empty"]))
    monkeypatch.setattr(eval_script, "load_config", _patched_load_config(runs_dir))

    assert eval_script.main() == 0


def test_exits_0_and_writes_one_report_on_synthetic_gold_set(
    built_index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    index_dir, chunks = built_index
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(_gold_line(chunks[0]) + "\n")
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(sys, "argv", _argv(index_dir, gold_path, runs_dir))
    monkeypatch.setattr(eval_script, "load_config", _patched_load_config(runs_dir))

    exit_code = eval_script.main()

    assert exit_code == 0
    report_files = list(runs_dir.glob("*.json"))
    assert len(report_files) == 1
    report = EvalReport.model_validate_json(report_files[0].read_text())
    assert report.num_queries == 1
    assert report.num_scored == 1


def test_exits_1_on_missing_index_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text("")
    runs_dir = tmp_path / "runs"
    missing_index_dir = tmp_path / "no_such_index"

    monkeypatch.setattr(sys, "argv", _argv(missing_index_dir, gold_path, runs_dir))
    monkeypatch.setattr(eval_script, "load_config", _patched_load_config(runs_dir))

    assert eval_script.main() == 1


def test_exits_1_on_missing_gold_path(built_index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    index_dir, _ = built_index
    runs_dir = tmp_path / "runs"
    missing_gold_path = tmp_path / "no_such_gold.jsonl"

    monkeypatch.setattr(sys, "argv", _argv(index_dir, missing_gold_path, runs_dir))
    monkeypatch.setattr(eval_script, "load_config", _patched_load_config(runs_dir))

    assert eval_script.main() == 1


def test_prints_min_queries_warn_warning(
    built_index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    index_dir, chunks = built_index
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(_gold_line(chunks[0]) + "\n")
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(sys, "argv", _argv(index_dir, gold_path, runs_dir))
    monkeypatch.setattr(eval_script, "load_config", _patched_load_config(runs_dir))

    eval_script.main()
    stderr = capsys.readouterr().err
    assert "min_queries_warn" in stderr


def test_json_flag_emits_parseable_json(
    built_index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    index_dir, chunks = built_index
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(_gold_line(chunks[0]) + "\n")
    runs_dir = tmp_path / "runs"

    monkeypatch.setattr(
        sys, "argv", _argv(index_dir, gold_path, runs_dir, ["--json"])
    )
    monkeypatch.setattr(eval_script, "load_config", _patched_load_config(runs_dir))

    eval_script.main()
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["num_queries"] == 1


def _patched_load_config(runs_dir: Path):
    """Build a load_config stand-in that redirects eval.runs_dir to `runs_dir`
    (everything else -- index_dir, gold_path -- comes from the CLI flags) and
    otherwise reads real config/<name>.yaml files (offline_smoke, eval)."""
    original_load_config = eval_script.load_config

    def _patched(name: str) -> dict:
        config = original_load_config(name)
        if name == "eval":
            config["eval"]["runs_dir"] = str(runs_dir)
        return config

    return _patched
