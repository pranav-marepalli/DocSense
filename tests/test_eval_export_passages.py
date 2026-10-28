"""Acceptance criterion §6.14: export_eval_passages.py links to check_eval_leakage.py."""

import sys
from pathlib import Path

import pytest

from docsense.index.build import build_index
from scripts import export_eval_passages as export_script
from scripts.check_eval_leakage import find_leaked_examples
from tests.conftest import write_sample_chunks


def _retriever_config() -> dict:
    return {
        "variant_name": "offline_smoke",
        "seed": 0,
        "record_format": "jsonl",
        "embedder": {
            "type": "hash",
            "model_name": "BAAI/bge-small-en-v1.5",
            "dim": 16,
            "normalize": True,
            "batch_size": 4,
            "device": None,
            "max_seq_length": 512,
            "query_prefix": "",
            "passage_prefix": "",
            "trust_remote_code": False,
        },
        "index": {"type": "numpy_flat_ip", "num_threads": 1},
        "retrieval": {"top_k": 5, "max_chunks_per_paper": None},
    }


@pytest.fixture
def built_index(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = write_sample_chunks(chunks_path)
    index_dir = tmp_path / "index"
    build_index(chunks_path, index_dir, _retriever_config())
    return index_dir, chunks


def test_export_and_leakage_check_link_end_to_end(
    built_index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    index_dir, chunks = built_index
    target = chunks[0]

    gold_path = tmp_path / "gold.jsonl"
    gold_line = (
        '{"query_id": "q001", "query": "what does chunk zero say", '
        '"query_type": "factoid", "difficulty": "easy", "unanswerable": false, '
        f'"spans": [{{"arxiv_id": "{target.arxiv_id}", "page_start": {target.page_start}, '
        f'"page_end": {target.page_end}, "relevance": 2, "note": null, '
        f'"source_chunk_id": "{target.chunk_id}"}}], "tags": [], "labeled_by": "tester", '
        '"labeled_at": "2024-01-01T00:00:00+00:00", "pooled_from": ["dense"], "pool_k": 20}'
    )
    gold_path.write_text(gold_line + "\n")

    out_path = tmp_path / "eval_texts.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_eval_passages.py",
            "--config",
            "eval",
            "--index-dir",
            str(index_dir),
            "--out",
            str(out_path),
        ],
    )

    original_load_config = export_script.load_config

    def _patched_load_config(name: str) -> dict:
        config = original_load_config(name)
        if name == "eval":
            config["eval"]["gold_path"] = str(gold_path)
            config["retriever_config"] = "offline_smoke"
        return config

    monkeypatch.setattr(export_script, "load_config", _patched_load_config)

    exit_code = export_script.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert out_path.exists()
    assert "check_eval_leakage.py" in output

    lines = [line for line in out_path.read_text().splitlines() if line.strip()]
    assert len(lines) >= 2  # at least the query and the overlapping-passage record
    assert any('"kind": "query"' in line for line in lines)
    assert any('"kind": "passage"' in line for line in lines)

    import json

    eval_texts = [json.loads(line)["text"] for line in lines]

    # A deliberately copied passage must be detected as leaked by check_eval_leakage.py.
    copied_passage = target.text + " " * 1 + "extra padding words to exceed the shingle size here"
    train_texts = [copied_passage, "a totally unrelated sentence about optimizer warmup schedules"]
    leaked = find_leaked_examples(eval_texts, train_texts, n=5)
    assert leaked == [0]


# --- R1: --out refuses to overwrite the configured gold set ---------------------


def test_main_refuses_out_path_equal_to_configured_gold_path(
    built_index, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """Reproduces the reviewer's empirical repro verbatim: pointing --out at
    the configured gold_path must exit non-zero and leave the gold file
    untouched, instead of silently unlink()-ing and replacing it."""
    index_dir, _ = built_index
    gold_path = tmp_path / "gold_set.jsonl"
    gold_path.write_text('{"query_id": "q001"}\n')  # stands in for real labels
    original_bytes = gold_path.read_bytes()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_eval_passages.py",
            "--config",
            "eval",
            "--index-dir",
            str(index_dir),
            "--out",
            str(gold_path),
        ],
    )

    original_load_config = export_script.load_config

    def _patched_load_config(name: str) -> dict:
        config = original_load_config(name)
        if name == "eval":
            config["eval"]["gold_path"] = str(gold_path)
            config["retriever_config"] = "offline_smoke"
        return config

    monkeypatch.setattr(export_script, "load_config", _patched_load_config)

    exit_code = export_script.main()
    stderr = capsys.readouterr().err

    assert exit_code != 0
    assert "gold" in stderr.lower()
    assert gold_path.read_bytes() == original_bytes
