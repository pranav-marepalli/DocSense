"""Acceptance criterion §6.12: scripts/label_eval.py, driven by a scripted
`prompt_fn` against a fixture index -- never real stdin.

Most of the pool-judging mechanics are exercised directly against
`run_labeling_session` (built from a fixture `DenseRetriever`, exactly like
tests/test_retrieve_dense.py). The CLI-level concerns (argument validation,
`--suggest-papers`) are exercised through `main()` with monkeypatched
`sys.argv` and `REPO_ROOT`/`load_config`, mirroring tests/test_scripts.py's
pattern for scripts/fetch_arxiv.py.
"""

import json
import sys
from pathlib import Path

import pytest

from docsense.eval.io import read_gold_set
from docsense.eval.schema import GoldExample, LabeledSpan
from docsense.index.build import build_index
from docsense.ingest.io import append_manifest, get_record_store
from docsense.ingest.schema import ManifestRecord
from docsense.retrieve.dense import DenseRetriever
from scripts import label_eval as label_eval_script
from tests.conftest import write_sample_chunks


def _retriever_config() -> dict:
    return {
        "variant_name": "test",
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
        "retrieval": {"top_k": 10, "max_chunks_per_paper": None},
    }


@pytest.fixture
def retriever(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = write_sample_chunks(chunks_path)
    index_dir = tmp_path / "index"
    config = _retriever_config()
    build_index(chunks_path, index_dir, config)
    return DenseRetriever.load(index_dir, config), chunks


def _scripted_prompt(answers: list[str]):
    """A `prompt_fn` that yields each of `answers` in turn, then raises
    StopIteration -- exercising `_SessionExhausted`'s handling, same as `q`."""
    it = iter(answers)

    def _prompt(_text: str) -> str:
        return next(it)

    return _prompt


def _counting_prompt(answers: list[str]):
    """Like `_scripted_prompt` but also records every prompt call count."""
    it = iter(answers)
    calls = {"n": 0}

    def _prompt(_text: str) -> str:
        calls["n"] += 1
        return next(it)

    return _prompt, calls


@pytest.fixture
def store():
    return get_record_store("jsonl")


# --- Core session mechanics (direct against run_labeling_session) --------------


def test_grading_two_candidates_then_n_then_q_appends_exactly_one_line(
    retriever, tmp_path: Path, store
):
    dense_retriever, chunks = retriever
    gold_path = tmp_path / "gold.jsonl"
    # Two queries: the session finalizes the first (grade 2, grade 1, then "n"
    # to stop early), picks query_type/difficulty, then hits "q" on the second
    # query's first candidate.
    prompt_fn = _scripted_prompt(["2", "1", "n", "table", "medium", "q"])

    appended = label_eval_script.run_labeling_session(
        retriever=dense_retriever,
        queries=[chunks[0].text, "a second query never finalized"],
        gold_path=gold_path,
        store=store,
        labeled_by="alice",
        pool_k=5,
        snippet_chars=400,
        prompt_fn=prompt_fn,
    )

    assert appended == 1
    examples = read_gold_set(gold_path, store)
    assert len(examples) == 1
    example = examples[0]
    assert example.labeled_by == "alice"
    assert example.pooled_from == ["dense"]
    assert example.pool_k == 5
    assert example.query_type == "table"
    assert example.difficulty == "medium"
    assert example.unanswerable is False
    assert len(example.spans) == 2
    assert {span.relevance for span in example.spans} == {1, 2}
    assert all(span.source_chunk_id is not None for span in example.spans)


def test_blank_and_garbage_input_reprompt_instead_of_defaulting(retriever, tmp_path: Path, store):
    dense_retriever, chunks = retriever
    gold_path = tmp_path / "gold.jsonl"
    prompt_fn, calls = _counting_prompt(
        ["", "garbage", "2", "n", "factoid", "nonsense", "easy"]
    )

    appended = label_eval_script.run_labeling_session(
        retriever=dense_retriever,
        queries=[chunks[0].text],
        gold_path=gold_path,
        store=store,
        labeled_by="alice",
        pool_k=3,
        snippet_chars=400,
        prompt_fn=prompt_fn,
    )

    assert appended == 1
    # 2 re-prompts (blank, garbage) + the "2" grade + "n" + "factoid" +
    # "nonsense" re-prompt + "easy" = 7 prompt_fn calls.
    assert calls["n"] == 7
    example = read_gold_set(gold_path, store)[0]
    assert example.spans[0].relevance == 2


def test_u_marks_unanswerable_with_no_relevant_span(retriever, tmp_path: Path, store):
    dense_retriever, chunks = retriever
    gold_path = tmp_path / "gold.jsonl"
    prompt_fn = _scripted_prompt(["u", "method", "hard"])

    appended = label_eval_script.run_labeling_session(
        retriever=dense_retriever,
        queries=[chunks[0].text],
        gold_path=gold_path,
        store=store,
        labeled_by="alice",
        pool_k=3,
        snippet_chars=400,
        prompt_fn=prompt_fn,
    )

    assert appended == 1
    example = read_gold_set(gold_path, store)[0]
    assert example.unanswerable is True
    assert all(span.relevance < 1 for span in example.spans)


def test_a_adds_manual_span_with_no_source_chunk_id(retriever, tmp_path: Path, store):
    dense_retriever, chunks = retriever
    gold_path = tmp_path / "gold.jsonl"
    prompt_fn = _scripted_prompt(["a 2301.00001:4-6 2", "n", "numeric_result", "hard"])

    appended = label_eval_script.run_labeling_session(
        retriever=dense_retriever,
        queries=[chunks[0].text],
        gold_path=gold_path,
        store=store,
        labeled_by="alice",
        pool_k=3,
        snippet_chars=400,
        prompt_fn=prompt_fn,
    )

    assert appended == 1
    example = read_gold_set(gold_path, store)[0]
    manual_spans = [s for s in example.spans if s.arxiv_id == "2301.00001" and s.page_start == 4]
    assert len(manual_spans) == 1
    manual = manual_spans[0]
    assert manual.page_end == 6
    assert manual.relevance == 2
    assert manual.source_chunk_id is None


def test_existing_line_is_byte_identical_and_already_present_query_is_skipped(
    retriever, tmp_path: Path, store
):
    dense_retriever, chunks = retriever
    gold_path = tmp_path / "gold.jsonl"

    # Seed the gold set with one example for chunks[0]'s text.
    seed_prompt = _scripted_prompt(["2", "n", "factoid", "easy"])
    label_eval_script.run_labeling_session(
        retriever=dense_retriever,
        queries=[chunks[0].text],
        gold_path=gold_path,
        store=store,
        labeled_by="alice",
        pool_k=3,
        snippet_chars=400,
        prompt_fn=seed_prompt,
    )
    original_bytes = gold_path.read_bytes()

    # A prompt_fn that would raise if ever called: the already-labeled query
    # must be skipped without asking anything.
    def _never_prompt(_text: str) -> str:
        raise AssertionError("prompt_fn must not be called for an already-labeled query")

    appended = label_eval_script.run_labeling_session(
        retriever=dense_retriever,
        queries=[chunks[0].text],  # same query text, same (normalized) form
        gold_path=gold_path,
        store=store,
        labeled_by="bob",
        pool_k=3,
        snippet_chars=400,
        prompt_fn=_never_prompt,
    )

    assert appended == 0
    assert gold_path.read_bytes() == original_bytes


def _seed_gold_example(query_id: str, query: str) -> dict:
    return GoldExample(
        query_id=query_id,
        query=query,
        query_type="factoid",
        difficulty="easy",
        unanswerable=False,
        spans=[
            LabeledSpan(
                arxiv_id="2301.00001",
                page_start=1,
                page_end=1,
                relevance=1,
                note=None,
                source_chunk_id=None,
            )
        ],
        tags=[],
        labeled_by="seed",
        labeled_at="2024-01-01T00:00:00+00:00",
        pooled_from=[],
        pool_k=None,
    ).model_dump(mode="json")


def test_next_query_id_fills_gap_from_max_suffix_not_len_existing(
    retriever, tmp_path: Path, store
):
    """R2 regression: existing ids q001, q002, q004 (a gap -- e.g. from a
    deleted/renumbered line) must produce q005 next, not `len(existing) + 1
    == 4` (which would collide with the existing q004 and raise inside
    `read_gold_set`'s duplicate-query_id check)."""
    dense_retriever, chunks = retriever
    gold_path = tmp_path / "gold.jsonl"
    seed_lines = [
        _seed_gold_example("q001", "first seeded query"),
        _seed_gold_example("q002", "second seeded query"),
        _seed_gold_example("q004", "fourth seeded query (a gap at q003)"),
    ]
    gold_path.write_text("\n".join(json.dumps(line) for line in seed_lines) + "\n")

    prompt_fn = _scripted_prompt(["2", "n", "factoid", "easy"])

    appended = label_eval_script.run_labeling_session(
        retriever=dense_retriever,
        queries=[chunks[0].text],
        gold_path=gold_path,
        store=store,
        labeled_by="alice",
        pool_k=3,
        snippet_chars=400,
        prompt_fn=prompt_fn,
    )

    assert appended == 1
    examples = read_gold_set(gold_path, store)
    assert [e.query_id for e in examples] == ["q001", "q002", "q004", "q005"]


def test_exhausting_prompt_sequence_without_completed_query_appends_nothing(
    retriever, tmp_path: Path, store
):
    dense_retriever, chunks = retriever
    gold_path = tmp_path / "gold.jsonl"
    prompt_fn = _scripted_prompt(["2"])  # grades one candidate, then runs out

    appended = label_eval_script.run_labeling_session(
        retriever=dense_retriever,
        queries=[chunks[0].text],
        gold_path=gold_path,
        store=store,
        labeled_by="alice",
        pool_k=3,
        snippet_chars=400,
        prompt_fn=prompt_fn,
    )

    assert appended == 0
    assert read_gold_set(gold_path, store) == []


def test_all_zero_grades_prints_explicit_skip_message_not_silent(
    retriever, tmp_path: Path, store, capsys: pytest.CaptureFixture
):
    """R5: grading every candidate 0 (genuinely "nothing in the pool was
    relevant") without marking the query unanswerable must not silently
    discard the human's judgments -- it must say so and point at `u`."""
    dense_retriever, chunks = retriever
    gold_path = tmp_path / "gold.jsonl"
    prompt_fn = _scripted_prompt(["0", "0", "0"])  # grades all 3 pooled candidates 0

    appended = label_eval_script.run_labeling_session(
        retriever=dense_retriever,
        queries=[chunks[0].text],
        gold_path=gold_path,
        store=store,
        labeled_by="alice",
        pool_k=3,
        snippet_chars=400,
        prompt_fn=prompt_fn,
    )

    stderr = capsys.readouterr().err
    assert appended == 0
    assert read_gold_set(gold_path, store) == []
    assert "skipping" in stderr.lower()
    assert "u" in stderr  # points at the unanswerable command


def test_skip_candidate_with_s_does_not_record_a_span(retriever, tmp_path: Path, store):
    dense_retriever, chunks = retriever
    gold_path = tmp_path / "gold.jsonl"
    prompt_fn = _scripted_prompt(["s", "2", "n", "comparison", "medium"])

    label_eval_script.run_labeling_session(
        retriever=dense_retriever,
        queries=[chunks[0].text],
        gold_path=gold_path,
        store=store,
        labeled_by="alice",
        pool_k=3,
        snippet_chars=400,
        prompt_fn=prompt_fn,
    )

    example = read_gold_set(gold_path, store)[0]
    assert len(example.spans) == 1
    assert example.spans[0].relevance == 2


# --- CLI-level concerns via main() ---------------------------------------------


def _patch_repo_root_and_config(monkeypatch, tmp_root: Path, overrides: dict):
    """Mirrors tests/test_scripts.py's pattern: monkeypatch REPO_ROOT and
    load_config so main() reads real config/<name>.yaml files but with paths
    (queries_path, gold_path, index_dir) redirected under `tmp_root`."""
    monkeypatch.setattr(label_eval_script, "REPO_ROOT", tmp_root)
    original_load_config = label_eval_script.load_config

    def _patched(name: str) -> dict:
        config = original_load_config(name)
        if name in overrides:
            for path_keys, value in overrides[name]:
                target = config
                for key in path_keys[:-1]:
                    target = target[key]
                target[path_keys[-1]] = value
        return config

    monkeypatch.setattr(label_eval_script, "load_config", _patched)


def test_missing_labeled_by_exits_nonzero_with_readme_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    monkeypatch.setattr(sys, "argv", ["label_eval.py", "--config", "eval"])

    exit_code = label_eval_script.main()
    stderr = capsys.readouterr().err

    assert exit_code != 0
    assert "README.md" in stderr


def test_missing_queries_path_exits_nonzero_with_readme_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    _patch_repo_root_and_config(
        monkeypatch,
        tmp_path,
        overrides={"eval": [(["labeling", "queries_path"], "definitely/missing/queries.txt")]},
    )
    monkeypatch.setattr(
        sys, "argv", ["label_eval.py", "--config", "eval", "--labeled-by", "alice"]
    )

    exit_code = label_eval_script.main()
    stderr = capsys.readouterr().err

    assert exit_code != 0
    assert "README.md" in stderr


def test_main_refuses_non_interactive_stdin_with_default_prompt_fn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """R6: `prompt_fn` defaults to `input`, which reads identically from a
    pipe or a terminal -- so `yes 2 | python scripts/label_eval.py ...` would
    otherwise be a de facto bulk-label path. `main()` must refuse to start
    the real interactive session when stdin is not a tty."""
    queries_path = tmp_path / "queries.txt"
    queries_path.write_text("a real question about the corpus\n")

    _patch_repo_root_and_config(
        monkeypatch,
        tmp_path,
        overrides={"eval": [(["labeling", "queries_path"], str(queries_path))]},
    )
    monkeypatch.setattr(sys, "stdin", type("_Stdin", (), {"isatty": lambda self: False})())
    monkeypatch.setattr(
        sys, "argv", ["label_eval.py", "--config", "eval", "--labeled-by", "alice"]
    )

    exit_code = label_eval_script.main()
    stderr = capsys.readouterr().err

    assert exit_code != 0
    assert "interactive" in stderr.lower()
    assert "README.md" in stderr


def test_main_with_injected_prompt_fn_proceeds_despite_non_tty_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The non-tty refusal (R6) is specifically about the real `input` path --
    a test (or any caller) that injects its own `prompt_fn` is not the
    de-facto-bulk-label risk the guard targets, so it must still work."""
    from docsense.config import load_config

    tmp_root = tmp_path / "repo"
    chunks_path = tmp_root / "data" / "parsed" / "chunks.jsonl"
    index_dir = tmp_root / "data" / "index" / "offline_smoke"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    built_chunks = write_sample_chunks(chunks_path)

    # Build the index with the real offline_smoke.yaml config (so the
    # embedder main() later loads for DenseRetriever.load matches exactly).
    real_offline_smoke_config = load_config("offline_smoke")
    build_index(chunks_path, index_dir, real_offline_smoke_config)

    queries_path = tmp_root / "data" / "eval" / "queries.txt"
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    queries_path.write_text(built_chunks[0].text + "\n")
    gold_path = tmp_root / "data" / "eval" / "gold_set.jsonl"

    _patch_repo_root_and_config(
        monkeypatch,
        tmp_root,
        overrides={
            "eval": [
                (["labeling", "queries_path"], str(queries_path)),
                (["eval", "gold_path"], str(gold_path)),
                (["retriever_config"], "offline_smoke"),
            ],
            "offline_smoke": [(["index_dir"], str(index_dir))],
        },
    )
    monkeypatch.setattr(sys, "stdin", type("_Stdin", (), {"isatty": lambda self: False})())
    monkeypatch.setattr(
        sys, "argv", ["label_eval.py", "--config", "eval", "--labeled-by", "alice"]
    )

    prompt_fn = _scripted_prompt(["2", "n", "factoid", "easy"])

    exit_code = label_eval_script.main(prompt_fn=prompt_fn)

    assert exit_code == 0
    store = get_record_store("jsonl")
    examples = read_gold_set(gold_path, store)
    assert len(examples) == 1


def test_suggest_papers_prints_n_rows_deterministic_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_store = get_record_store("jsonl")
    for i in range(10):
        append_manifest(
            ManifestRecord(
                arxiv_id=f"2301.0000{i}",
                version=1,
                title=f"Paper {i}",
                categories=["cs.CL"],
                query_category="cs.CL",
                published="2023-01-01T00:00:00Z",
                pdf_path=f"data/raw/2301.0000{i}v1.pdf",
                sha256="a" * 64,
                num_bytes=1,
                fetched_at="2024-01-01T00:00:00+00:00",
            ),
            manifest_path,
            manifest_store,
        )

    _patch_repo_root_and_config(
        monkeypatch,
        tmp_path,
        overrides={"ingest": [(["manifest_path"], "manifest.jsonl")]},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["label_eval.py", "--config", "eval", "--suggest-papers", "3"],
    )

    exit_code = label_eval_script.main()
    first_output = capsys.readouterr().out

    exit_code_2 = label_eval_script.main()
    second_output = capsys.readouterr().out

    assert exit_code == 0
    assert exit_code_2 == 0
    assert len(first_output.splitlines()) == 3
    assert first_output == second_output  # deterministic under the configured seed
    assert not (tmp_path / "data" / "eval" / "gold_set.jsonl").exists()
