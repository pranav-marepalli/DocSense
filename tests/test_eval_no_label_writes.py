"""Acceptance criterion §6.11: fabrication guards.

CLAUDE.md's hard rule -- "Never create, edit, or fabricate evaluation labels
or results. The human labels the gold set." -- is load-bearing for this test
module.

R3 (Phase 3 review): the original version of this module guarded
`write_eval_report` behaviorally but caught everything else only with a
narrow source-text scan -- a literal "gold_set" string appearing on the same
line as `open(..., "w")` or `write_text`. That is exactly why it missed R1:
`export_eval_texts` reached a gold-shaped path through a `Path` *parameter*
(never the literal string `"gold_set"` in source), and wrote via `unlink()`
followed by `store.append(...)`, neither of which the old regex matched at
all. This module now leads with *behavioral* tests -- every path-accepting
writer in `docsense.eval.io`, plus the `export_eval_passages.py` script,
driven with the real configured gold path and asserted to raise/refuse -- and
keeps the structural/name scan only as defense in depth, broadened to also
catch append mode, `RecordStore.append(...)`, and `Path.unlink()`.
"""

import ast
import json
import re
import sys
from pathlib import Path

import pytest

from docsense.eval.io import export_eval_texts, write_eval_report
from docsense.eval.schema import EvalReport, RetrievedRef
from docsense.ingest.io import get_record_store

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SRC_DIR = REPO_ROOT / "src" / "docsense" / "eval"

_FORBIDDEN_NAME_RE = re.compile(r"write_gold|append_gold|generate_gold|synthes", re.IGNORECASE)


def _eval_source_files() -> list[Path]:
    return sorted(EVAL_SRC_DIR.glob("*.py"))


def _defined_names(tree: ast.Module) -> set[str]:
    """Every function, class, and assigned-variable name defined at any scope.

    Prose in a docstring (e.g. "there is no write_gold_set function") does not
    count -- the acceptance criterion is about a *name that exists*, not a
    string that mentions one.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def test_no_forbidden_writer_name_exists_in_eval_package():
    for path in _eval_source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for name in _defined_names(tree):
            assert not _FORBIDDEN_NAME_RE.search(name), f"forbidden name {name!r} defined in {path}"


# --- R3: broadened structural scan (defense in depth, not the primary guard) ----

# Broadened from the original "w"/write_text-only scan to also catch append
# mode ("a"/"x"), RecordStore.append(...), and Path.unlink() -- the exact
# write primitives R1's export_eval_texts used that the narrow version missed.
_WRITE_PRIMITIVE_RE = re.compile(
    r'open\([^)]*["\'][wax]["\']'  # open(..., "w"/"a"/"x")
    r"|write_text\("
    r"|write_bytes\("
    r"|\.unlink\("
    r"|\.append\(",
)


def test_no_module_under_docsense_writes_to_gold_set_as_a_target():
    """Source scan for any write primitive on a line mentioning "gold_set"."""
    docsense_src = REPO_ROOT / "src" / "docsense"
    offending: list[str] = []
    for path in docsense_src.rglob("*.py"):
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "gold_set" not in line:
                continue
            if _WRITE_PRIMITIVE_RE.search(line):
                offending.append(f"{path}:{lineno}: {line.strip()}")
    assert offending == [], f"found gold_set write targets: {offending}"


# --- R3(b): behavioral guards -- every path-accepting writer in eval.io, -----
# --- plus the export-passages script, must refuse the real gold path. -------


def _report(gold_path: str, run_id: str) -> EvalReport:
    return EvalReport(
        run_id=run_id,
        started_at="2024-01-01T00:00:00+00:00",
        finished_at="2024-01-01T00:00:01+00:00",
        gold_path=gold_path,
        gold_sha256="a" * 64,
        num_queries=0,
        num_scored=0,
        num_unanswerable=0,
        retriever="dense",
        retriever_config="baseline",
        variant_name="baseline",
        embedder_model="hash",
        index_fingerprint="b" * 64,
        k_values=[1, 3, 5, 10],
        primary_k=5,
        binary_relevance_threshold=1,
        retrieval_top_k=10,
        seed=0,
        library_versions={},
        metrics={},
        metrics_by_slice={},
        mean_latency_ms=None,
        per_query=[],
    )


def test_write_eval_report_raises_for_gold_path_runs_dir(tmp_path: Path):
    gold_path = tmp_path / "gold_set.jsonl"
    report = _report(gold_path=str(gold_path), run_id="gold_set")

    with pytest.raises(ValueError):
        write_eval_report(report, tmp_path)


def test_export_eval_texts_raises_when_out_path_is_the_configured_gold_path(tmp_path: Path):
    """The R1 repro, directly against the library function: `out_path` equal
    to the configured gold path must raise, not unlink()-and-overwrite it."""
    gold_path = tmp_path / "gold_set.jsonl"
    gold_path.write_text('{"query_id": "q001"}\n')
    original_bytes = gold_path.read_bytes()
    store = get_record_store("jsonl")

    with pytest.raises(ValueError):
        export_eval_texts([], [], gold_path, store, gold_path=gold_path)

    assert gold_path.read_bytes() == original_bytes


def test_export_eval_texts_raises_for_any_gold_named_out_path(tmp_path: Path):
    """The name heuristic alone (no `gold_path` argument) must also refuse."""
    out_path = tmp_path / "my_gold_export.jsonl"
    store = get_record_store("jsonl")

    with pytest.raises(ValueError):
        export_eval_texts([], [], out_path, store)

    assert not out_path.exists()


def test_export_eval_passages_script_refuses_the_reviewers_repro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """Reproduces the reviewer's exact empirical repro end to end through the
    script: `--out data/eval/gold_set.jsonl` must exit non-zero and leave the
    gold file untouched, never unlink() + rewrite it with derived
    (query_id, text) records."""
    from docsense.index.build import build_index
    from scripts import export_eval_passages as export_script
    from tests.conftest import write_sample_chunks

    chunks_path = tmp_path / "chunks.jsonl"
    chunks = write_sample_chunks(chunks_path)
    index_dir = tmp_path / "index"
    retriever_config = {
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
    build_index(chunks_path, index_dir, retriever_config)

    gold_path = tmp_path / "gold_set.jsonl"
    gold_path.write_text(
        json.dumps(
            {
                "query_id": "q001",
                "query": "a real human-labeled query",
                "query_type": "factoid",
                "difficulty": "easy",
                "unanswerable": False,
                "spans": [
                    {
                        "arxiv_id": chunks[0].arxiv_id,
                        "page_start": chunks[0].page_start,
                        "page_end": chunks[0].page_end,
                        "relevance": 2,
                        "note": None,
                        "source_chunk_id": chunks[0].chunk_id,
                    }
                ],
                "tags": [],
                "labeled_by": "tester",
                "labeled_at": "2024-01-01T00:00:00+00:00",
                "pooled_from": ["dense"],
                "pool_k": 20,
            }
        )
        + "\n"
    )
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


def test_retrieved_ref_has_no_text_field():
    assert "text" not in RetrievedRef.model_fields
