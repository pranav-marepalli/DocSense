"""Acceptance criterion 14: scripts/ingest.py and scripts/fetch_arxiv.py --dry-run."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pymupdf as fitz
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    """Load scripts/<name>.py as a standalone module (scripts/ is not a package)."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"docsense_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_valid_pdf(path: Path, body: str) -> None:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), body)
    doc.save(path)
    doc.close()


def test_ingest_script_exits_zero_when_all_succeed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pdf_dir = tmp_path / "raw"
    pdf_dir.mkdir()
    _make_valid_pdf(pdf_dir / "2301.00001.pdf", "Some paper content. " * 40)
    _make_valid_pdf(pdf_dir / "2301.00002.pdf", "More paper content. " * 40)
    out_dir = tmp_path / "parsed"

    ingest_script = _load_script("ingest")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest.py",
            "--config",
            "ingest",
            "--pdf-dir",
            str(pdf_dir),
            "--out-dir",
            str(out_dir),
        ],
    )
    exit_code = ingest_script.main()
    assert exit_code == 0
    assert (out_dir / "ingest_report.json").exists()


def test_ingest_script_exits_nonzero_when_a_paper_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pdf_dir = tmp_path / "raw"
    pdf_dir.mkdir()
    _make_valid_pdf(pdf_dir / "2301.00001.pdf", "Some paper content. " * 40)
    (pdf_dir / "2301.00002.pdf").write_bytes(b"%PDF-1.4 not a real pdf")
    out_dir = tmp_path / "parsed"

    ingest_script = _load_script("ingest")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest.py",
            "--config",
            "ingest",
            "--pdf-dir",
            str(pdf_dir),
            "--out-dir",
            str(out_dir),
        ],
    )
    exit_code = ingest_script.main()
    assert exit_code != 0


def test_ingest_script_default_manifest_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """R5: scripts/ingest.py's default mode (neither --manifest nor --pdf-dir given)
    reads data/manifest.jsonl -- the canonical path (§3) -- not --pdf-dir."""
    from docsense.ingest import pipeline as pipeline_module
    from docsense.ingest.io import append_manifest, get_record_store
    from docsense.ingest.schema import ManifestRecord

    fake_repo_root = tmp_path
    monkeypatch.setattr(pipeline_module, "REPO_ROOT", fake_repo_root)

    pdf_dir = fake_repo_root / "data" / "raw"
    pdf_dir.mkdir(parents=True)
    _make_valid_pdf(pdf_dir / "2301.00001.pdf", "Some paper content. " * 40)

    manifest_path = fake_repo_root / "data" / "manifest.jsonl"
    store = get_record_store("jsonl")
    append_manifest(
        ManifestRecord(
            arxiv_id="2301.00001",
            version=1,
            title="T",
            categories=["cs.CL"],
            query_category="cs.CL",
            published="2023-01-10T09:00:00Z",
            pdf_path="data/raw/2301.00001.pdf",
            sha256="a" * 64,
            num_bytes=1,
            fetched_at="2024-01-01T00:00:00+00:00",
        ),
        manifest_path,
        store,
    )

    out_dir = tmp_path / "parsed"
    ingest_script = _load_script("ingest")
    monkeypatch.setattr(ingest_script, "REPO_ROOT", fake_repo_root)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ingest.py", "--config", "ingest", "--out-dir", str(out_dir)],
    )
    exit_code = ingest_script.main()

    assert exit_code == 0
    assert (out_dir / "papers" / "2301.00001.json").exists()


def test_fetch_arxiv_dry_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fetch_script = _load_script("fetch_arxiv")

    from docsense.ingest import fetch_arxiv as fetch_arxiv_module

    sample_feed = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">\n'
        "  <entry>\n"
        "    <id>http://arxiv.org/abs/2301.00001v1</id>\n"
        "    <updated>2023-01-10T09:00:00Z</updated>\n"
        "    <published>2023-01-10T09:00:00Z</published>\n"
        "    <title>Dry Run Paper</title>\n"
        '    <link title="pdf" href="http://arxiv.org/pdf/2301.00001v1" rel="related" '
        'type="application/pdf"/>\n'
        '    <arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>\n'
        '    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>\n'
        "  </entry>\n"
        "</feed>"
    )

    def _fake_http_get(url: str, timeout_s: float) -> bytes:
        if "api/query" in url:
            return sample_feed.encode("utf-8")
        raise AssertionError("dry-run must never download a PDF")

    monkeypatch.setattr(fetch_arxiv_module, "_http_get", _fake_http_get)

    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"

    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_arxiv.py", "--config", "ingest", "--max-papers", "1", "--dry-run"],
    )
    monkeypatch.setattr(fetch_script, "REPO_ROOT", tmp_path)

    # load_config reads from the real config/ dir (CONFIG_DIR is relative to the
    # installed package), so point manifest_path/pdf_dir overrides via config instead.
    original_load_config = fetch_script.load_config

    def _patched_load_config(name: str) -> dict:
        config = original_load_config(name)
        config["manifest_path"] = "manifest.jsonl"
        config["pdf_dir"] = "raw"
        config["fetch"]["categories"] = ["cs.CL"]
        return config

    monkeypatch.setattr(fetch_script, "load_config", _patched_load_config)

    exit_code = fetch_script.main()

    assert exit_code == 0
    assert not manifest_path.exists()
    assert not pdf_dir.exists() or list(pdf_dir.glob("*")) == []


def test_fetch_arxiv_dry_run_reports_rebuild_candidates_separately_from_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """R19: on a fresh clone, a full manifest with no PDFs on disk is exactly the
    case the rebuild-from-manifest flow matters most for. `--dry-run` must report
    those as rebuild-needed (separately from new-paper candidates), not silently
    report 0 because it only looked at `iter_new_candidates`."""
    import hashlib

    from docsense.ingest import fetch_arxiv as fetch_arxiv_module
    from docsense.ingest.io import append_manifest, get_record_store
    from docsense.ingest.schema import ManifestRecord

    fetch_script = _load_script("fetch_arxiv")

    def _fail_any_request(url: str, timeout_s: float) -> bytes:
        raise AssertionError(f"dry-run must not make network requests, got {url}")

    monkeypatch.setattr(fetch_arxiv_module, "_http_get", _fail_any_request)

    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"  # no PDFs written -- simulates a fresh clone
    store = get_record_store("jsonl")

    for i in range(3):
        arxiv_id = f"2301.0005{i}"
        append_manifest(
            ManifestRecord(
                arxiv_id=arxiv_id,
                version=1,
                title=f"Paper {arxiv_id}",
                categories=["cs.CL"],
                query_category="cs.CL",
                published="2023-01-10T09:00:00Z",
                pdf_path=f"data/raw/{arxiv_id}v1.pdf",
                sha256=hashlib.sha256(arxiv_id.encode()).hexdigest(),
                num_bytes=1,
                fetched_at="2023-01-01T00:00:00+00:00",
            ),
            manifest_path,
            store,
        )

    monkeypatch.setattr(
        sys,
        "argv",
        # max_papers=0 so the new-candidate discovery pass (which would otherwise
        # make network requests) offers nothing; the rebuild pass makes none either.
        ["fetch_arxiv.py", "--config", "ingest", "--max-papers", "0", "--dry-run"],
    )
    monkeypatch.setattr(fetch_script, "REPO_ROOT", tmp_path)

    original_load_config = fetch_script.load_config

    def _patched_load_config(name: str) -> dict:
        config = original_load_config(name)
        config["manifest_path"] = "manifest.jsonl"
        config["pdf_dir"] = "raw"
        config["fetch"]["categories"] = ["cs.CL"]
        return config

    monkeypatch.setattr(fetch_script, "load_config", _patched_load_config)

    exit_code = fetch_script.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert len(manifest_path.read_text().splitlines()) == 3  # dry-run wrote nothing
    assert not pdf_dir.exists() or list(pdf_dir.glob("*")) == []
    assert "3 manifest paper(s) would be re-downloaded" in output
    assert "0 new paper(s) would be downloaded" in output
