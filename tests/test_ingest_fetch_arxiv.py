"""Acceptance criteria 1-5 and 15 (no real network, no real 3s sleep).

Every test in this file monkeypatches `docsense.ingest.fetch_arxiv._http_get`,
the module's single network entry point, so nothing here ever makes a real
request to arXiv.
"""

import re
import time
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

from docsense.ingest import fetch_arxiv
from docsense.ingest.fetch_arxiv import (
    RateLimiter,
    build_search_query,
    category_targets,
    download_pdf,
    fetch_corpus,
    fetch_result_page,
    iter_new_candidates,
    iter_rebuild_candidates,
    parse_atom_feed,
    sha256_file,
)
from docsense.ingest.io import get_record_store, read_manifest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_feed(entries: list[dict[str, Any]]) -> str:
    """Build a minimal valid Atom feed string for the given entry dicts."""
    entry_blocks = []
    for e in entries:
        categories_xml = "".join(
            f'<category term="{c}" scheme="http://arxiv.org/schemas/atom"/>'
            for c in e["categories"]
        )
        entry_blocks.append(
            f"""
  <entry>
    <id>http://arxiv.org/abs/{e["arxiv_id"]}v{e["version"]}</id>
    <updated>{e["published"]}</updated>
    <published>{e["published"]}</published>
    <title>{e["title"]}</title>
    <link title="pdf" href="{e["pdf_url"]}" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="{e["categories"][0]}" scheme="http://arxiv.org/schemas/atom"/>
    {categories_xml}
  </entry>"""
        )
    body = "".join(entry_blocks)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">'
        f"{body}\n</feed>"
    )


class FakeArxivTransport:
    """Fakes `_http_get` for both the Atom API and PDF downloads, offline."""

    def __init__(
        self,
        feeds_by_category: dict[str, list[str]],
        pdf_bodies: dict[str, bytes | Exception],
    ) -> None:
        self.feeds_by_category = feeds_by_category
        self.pdf_bodies = pdf_bodies
        self.urls: list[str] = []

    def __call__(self, url: str, timeout_s: float) -> bytes:
        self.urls.append(url)
        if url.startswith(fetch_arxiv.ARXIV_API_URL):
            params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            search_query = params["search_query"][0]
            start = int(params["start"][0])
            page_size = int(params["max_results"][0])
            category = re.search(r"cat:(\S+)", search_query).group(1)
            page_index = start // page_size
            pages = self.feeds_by_category.get(category, [])
            if page_index < len(pages):
                return pages[page_index].encode("utf-8")
            return _make_feed([]).encode("utf-8")

        body = self.pdf_bodies[url]
        if isinstance(body, Exception):
            raise body
        return body


def _entry(arxiv_id: str, categories: list[str], version: int = 1) -> dict[str, Any]:
    return {
        "arxiv_id": arxiv_id,
        "version": version,
        "title": f"Paper {arxiv_id}",
        "categories": categories,
        "published": "2023-01-10T09:00:00Z",
        "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}v{version}",
    }


def _base_fetch_config(**overrides: Any) -> dict:
    config: dict[str, Any] = {
        "record_format": "jsonl",
        "fetch": {
            "categories": ["cs.CL"],
            "published_after": "2023-01-01",
            "published_before": None,
            "max_papers": 3,
            "request_delay_seconds": 0.0,
            "page_size": 100,
            "timeout_seconds": 5.0,
            "max_retries": 2,
        },
    }
    config["fetch"].update(overrides)
    return config


# --- acceptance 1 -----------------------------------------------------------


def test_build_search_query_basic():
    query = build_search_query("cs.CL", "2023-01-01", "2023-06-01")
    assert query == "cat:cs.CL AND submittedDate:[202301010000 TO 202306012359]"


def test_build_search_query_none_before_uses_today():
    query = build_search_query("cs.CL", "2023-01-01", None)
    assert query.startswith("cat:cs.CL AND submittedDate:[202301010000 TO ")
    assert query.endswith("2359]")


def test_build_search_query_raises_on_empty_category():
    with pytest.raises(ValueError):
        build_search_query("", "2023-01-01", None)


def test_build_search_query_raises_on_bad_date():
    with pytest.raises(ValueError):
        build_search_query("cs.CL", "not-a-date", None)


def test_category_targets_even_split():
    targets = category_targets(["cs.CL", "cs.LG", "cs.CV", "cs.IR"], 10)
    assert targets == {"cs.CL": 3, "cs.LG": 3, "cs.CV": 2, "cs.IR": 2}
    assert sum(targets.values()) == 10
    assert max(targets.values()) - min(targets.values()) <= 1


def test_category_targets_raises_on_empty_categories():
    with pytest.raises(ValueError):
        category_targets([], 10)


# --- acceptance 3 ------------------------------------------------------------


def test_rate_limiter_enforces_gap():
    limiter = RateLimiter(min_interval_s=0.05)
    start = time.monotonic()
    limiter.wait()
    limiter.wait()
    limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.1


# --- acceptance 2 ------------------------------------------------------------


def test_parse_atom_feed_fixture():
    xml_text = (FIXTURES_DIR / "arxiv_feed_sample.xml").read_text()
    entries = parse_atom_feed(xml_text)

    assert len(entries) == 2
    first = entries[0]
    assert first["arxiv_id"] == "2301.12345"
    assert first["version"] == 1
    assert first["title"] == "A Great Paper About Testing"
    assert first["categories"] == ["cs.CL", "cs.LG"]
    assert first["published"] == "2023-01-10T09:00:00Z"
    assert first["pdf_url"] == "http://arxiv.org/pdf/2301.12345v1"

    second = entries[1]
    assert second["arxiv_id"] == "2302.54321"
    assert second["version"] == 2


def test_parse_atom_feed_entry_less_feed_returns_empty_list():
    xml_text = _make_feed([])
    assert parse_atom_feed(xml_text) == []


# --- sha256 / download_pdf ---------------------------------------------------


def test_sha256_file(tmp_path: Path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"some bytes")
    import hashlib

    assert sha256_file(path) == hashlib.sha256(b"some bytes").hexdigest()


def test_download_pdf_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    body = b"%PDF-1.4 fake pdf content"
    monkeypatch.setattr(fetch_arxiv, "_http_get", lambda url, timeout_s: body)

    dest = tmp_path / "2301.00001v1.pdf"
    sha256, num_bytes = download_pdf("http://example/pdf", dest, 5.0, RateLimiter(0.0))

    assert dest.exists()
    assert not dest.with_suffix(".part").exists()
    assert num_bytes == len(body)

    import hashlib

    assert sha256 == hashlib.sha256(body).hexdigest()


def test_download_pdf_rejects_non_pdf_body_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(fetch_arxiv, "_http_get", lambda url, timeout_s: b"not a pdf")

    dest = tmp_path / "2301.00002v1.pdf"
    with pytest.raises(ValueError):
        download_pdf("http://example/pdf", dest, 5.0, RateLimiter(0.0))

    assert not dest.exists()
    assert not dest.with_suffix(".part").exists()


def test_fetch_result_page_retries_then_raises(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def _always_fail(url: str, timeout_s: float) -> bytes:
        calls.append(url)
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(fetch_arxiv, "_http_get", _always_fail)
    limiter = RateLimiter(0.0)

    with pytest.raises(urllib.error.URLError):
        fetch_result_page("cat:cs.CL", 0, 100, 5.0, limiter, max_retries=3)

    assert len(calls) == 3


# --- acceptance 4: resumable --------------------------------------------------


def test_fetch_corpus_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    entries = [
        _entry("2301.00001", ["cs.CL"]),
        _entry("2301.00002", ["cs.CL"]),
        _entry("2301.00003", ["cs.CL"]),
    ]
    feed = _make_feed(entries)
    pdf_bodies = {e["pdf_url"]: b"%PDF-1.4 body for " + e["arxiv_id"].encode() for e in entries}

    transport = FakeArxivTransport(feeds_by_category={"cs.CL": [feed]}, pdf_bodies=pdf_bodies)
    monkeypatch.setattr(fetch_arxiv, "_http_get", transport)

    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"
    store = get_record_store("jsonl")

    from docsense.ingest.io import append_manifest
    from docsense.ingest.schema import ManifestRecord

    # Its PDF already exists on disk, so the rebuild/verify pass (decision 2, R2)
    # must not try to re-download it.
    pdf_dir.mkdir(parents=True)
    pre_existing_body = b"%PDF-1.4 pre-existing body"
    (pdf_dir / "2301.00001v1.pdf").write_bytes(pre_existing_body)
    import hashlib

    pre_existing = ManifestRecord(
        arxiv_id="2301.00001",
        version=1,
        title="Paper 2301.00001",
        categories=["cs.CL"],
        query_category="cs.CL",
        published="2023-01-10T09:00:00Z",
        pdf_path="raw/2301.00001v1.pdf",
        sha256=hashlib.sha256(pre_existing_body).hexdigest(),
        num_bytes=len(pre_existing_body),
        fetched_at="2023-01-01T00:00:00+00:00",
    )
    append_manifest(pre_existing, manifest_path, store)
    pre_existing_line = manifest_path.read_text().splitlines()[0]

    config = _base_fetch_config(max_papers=3)
    report = fetch_corpus(config, manifest_path, pdf_dir)

    assert report.newly_fetched == 2
    assert report.newly_fetched_by_category["cs.CL"] == 2
    assert report.already_in_manifest == 1
    assert not report.failed

    records = read_manifest(manifest_path, store)
    assert len(records) == 3
    assert manifest_path.read_text().splitlines()[0] == pre_existing_line

    # second run downloads nothing: target already met
    transport.urls.clear()
    second_report = fetch_corpus(config, manifest_path, pdf_dir)
    assert second_report.newly_fetched == 0
    assert read_manifest(manifest_path, store) == records
    assert transport.urls == []  # no requests at all: target already satisfied


def test_fetch_corpus_download_failure_lands_in_failed_and_leaves_no_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    entries = [_entry("2301.00010", ["cs.CL"]), _entry("2301.00011", ["cs.CL"])]
    feed = _make_feed(entries)
    pdf_bodies: dict[str, bytes | Exception] = {
        entries[0]["pdf_url"]: urllib.error.URLError("mid-run failure"),
        entries[1]["pdf_url"]: b"%PDF-1.4 ok body",
    }
    transport = FakeArxivTransport(feeds_by_category={"cs.CL": [feed]}, pdf_bodies=pdf_bodies)
    monkeypatch.setattr(fetch_arxiv, "_http_get", transport)

    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"
    config = _base_fetch_config(max_papers=2)

    report = fetch_corpus(config, manifest_path, pdf_dir)

    assert len(report.failed) == 1
    assert report.failed[0]["arxiv_id"] == "2301.00010"
    assert not (pdf_dir / "2301.00010v1.pdf").exists()
    assert not (pdf_dir / "2301.00010v1.part").exists()
    assert report.newly_fetched == 1


# --- acceptance 5: stratification --------------------------------------------


def test_fetch_corpus_stratifies_and_dedupes_cross_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    categories = ["cs.CL", "cs.LG", "cs.CV", "cs.IR"]

    cross_listed = _entry("2301.00099", ["cs.CL", "cs.LG"])
    cl_unique = _entry("2301.00001", ["cs.CL"])
    lg_unique = _entry("2301.00002", ["cs.LG"])
    cv_entries = [_entry("2301.00003", ["cs.CV"]), _entry("2301.00004", ["cs.CV"])]
    ir_entries = [_entry("2301.00005", ["cs.IR"]), _entry("2301.00006", ["cs.IR"])]

    feeds_by_category = {
        "cs.CL": [_make_feed([cl_unique, cross_listed])],
        "cs.LG": [_make_feed([cross_listed, lg_unique]), _make_feed([])],
        "cs.CV": [_make_feed(cv_entries)],
        "cs.IR": [_make_feed(ir_entries)],
    }
    all_entries = [cross_listed, cl_unique, lg_unique, *cv_entries, *ir_entries]
    pdf_bodies = {e["pdf_url"]: b"%PDF-1.4 body " + e["arxiv_id"].encode() for e in all_entries}

    transport = FakeArxivTransport(feeds_by_category=feeds_by_category, pdf_bodies=pdf_bodies)
    monkeypatch.setattr(fetch_arxiv, "_http_get", transport)

    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"
    config = _base_fetch_config(categories=categories, max_papers=8)

    report = fetch_corpus(config, manifest_path, pdf_dir)

    assert len(report.search_queries) == 4
    assert set(report.search_queries.keys()) == set(categories)
    assert report.category_targets == {"cs.CL": 2, "cs.LG": 2, "cs.CV": 2, "cs.IR": 2}

    store = get_record_store("jsonl")
    records = read_manifest(manifest_path, store)
    assert len([r for r in records if r.arxiv_id == "2301.00099"]) == 1  # downloaded exactly once
    assert report.newly_fetched < report.max_papers  # cs.LG falls short, no backfill

    assert report.newly_fetched_by_category["cs.CL"] == 2
    assert report.newly_fetched_by_category["cs.CV"] == 2
    assert report.newly_fetched_by_category["cs.IR"] == 2
    assert report.newly_fetched_by_category["cs.LG"] == 1  # exhausted early, under target


# --- R1: manifest pdf_path must be relative to the repo root ----------------


def test_fetch_corpus_manifest_pdf_path_is_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`ManifestRecord.pdf_path` must be relative to the repo root (schema contract),
    never a machine-specific absolute path."""
    entry = _entry("2301.00070", ["cs.CL"])
    feed = _make_feed([entry])
    pdf_bodies = {entry["pdf_url"]: b"%PDF-1.4 body"}

    transport = FakeArxivTransport(feeds_by_category={"cs.CL": [feed]}, pdf_bodies=pdf_bodies)
    monkeypatch.setattr(fetch_arxiv, "_http_get", transport)

    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"
    config = _base_fetch_config(max_papers=1)
    fetch_corpus(config, manifest_path, pdf_dir)

    store = get_record_store("jsonl")
    [record] = read_manifest(manifest_path, store)
    assert not Path(record.pdf_path).is_absolute()


# --- R2: rebuild path (decision 2) -------------------------------------------


def test_fetch_corpus_rebuilds_missing_pdf_and_flags_sha256_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """On a fresh clone the PDFs behind existing manifest records are missing; a
    missing PDF is re-downloaded at its recorded version, and a sha256 mismatch is a
    per-paper failure rather than a silent substitution (decision 2)."""
    import hashlib

    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"
    store = get_record_store("jsonl")

    from docsense.ingest.io import append_manifest
    from docsense.ingest.schema import ManifestRecord

    good_body = b"%PDF-1.4 good body"
    good_record = ManifestRecord(
        arxiv_id="2301.00050",
        version=1,
        title="Good Paper",
        categories=["cs.CL"],
        query_category="cs.CL",
        published="2023-01-10T09:00:00Z",
        pdf_path="data/raw/2301.00050v1.pdf",
        sha256=hashlib.sha256(good_body).hexdigest(),
        num_bytes=len(good_body),
        fetched_at="2023-01-01T00:00:00+00:00",
    )
    bad_record = ManifestRecord(
        arxiv_id="2301.00051",
        version=1,
        title="Bad Paper",
        categories=["cs.CL"],
        query_category="cs.CL",
        published="2023-01-10T09:00:00Z",
        pdf_path="data/raw/2301.00051v1.pdf",
        sha256="f" * 64,  # will not match whatever the "re-download" returns
        num_bytes=10,
        fetched_at="2023-01-01T00:00:00+00:00",
    )
    append_manifest(good_record, manifest_path, store)
    append_manifest(bad_record, manifest_path, store)
    # neither PDF exists on disk yet -- simulates a fresh clone

    rebuild_bodies = {
        "https://arxiv.org/pdf/2301.00050v1": good_body,
        "https://arxiv.org/pdf/2301.00051v1": b"%PDF-1.4 a different body entirely",
    }

    def _fake_http_get(url: str, timeout_s: float) -> bytes:
        return rebuild_bodies[url]

    monkeypatch.setattr(fetch_arxiv, "_http_get", _fake_http_get)

    # max_papers=0 so the "discover new candidates" pass makes no requests at all --
    # this test is only exercising the rebuild/verify pass.
    config = _base_fetch_config(categories=["cs.CL"], max_papers=0)
    report = fetch_corpus(config, manifest_path, pdf_dir)

    assert (pdf_dir / "2301.00050v1.pdf").exists()  # rebuilt, sha256 matched
    assert not (pdf_dir / "2301.00051v1.pdf").exists()  # mismatch: cleaned up, not kept
    assert len(report.failed) == 1
    assert report.failed[0]["arxiv_id"] == "2301.00051"
    assert "sha256" in report.failed[0]["error"].lower()


def test_fetch_corpus_rebuild_skips_pdfs_already_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A manifest record whose PDF is already present on disk with a matching
    sha256 is left alone: no re-download, no network request."""
    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"
    pdf_dir.mkdir(parents=True)
    store = get_record_store("jsonl")

    import hashlib

    from docsense.ingest.io import append_manifest
    from docsense.ingest.schema import ManifestRecord

    body = b"%PDF-1.4 already here"
    (pdf_dir / "2301.00060v1.pdf").write_bytes(body)
    record = ManifestRecord(
        arxiv_id="2301.00060",
        version=1,
        title="Already Here",
        categories=["cs.CL"],
        query_category="cs.CL",
        published="2023-01-10T09:00:00Z",
        pdf_path="data/raw/2301.00060v1.pdf",
        sha256=hashlib.sha256(body).hexdigest(),
        num_bytes=len(body),
        fetched_at="2023-01-01T00:00:00+00:00",
    )
    append_manifest(record, manifest_path, store)

    def _fail_any_request(url: str, timeout_s: float) -> bytes:
        raise AssertionError(f"no request should be made, got {url}")

    monkeypatch.setattr(fetch_arxiv, "_http_get", _fail_any_request)

    config = _base_fetch_config(categories=["cs.CL"], max_papers=0)
    report = fetch_corpus(config, manifest_path, pdf_dir)
    assert not report.failed
    assert (pdf_dir / "2301.00060v1.pdf").read_bytes() == body  # untouched


def test_fetch_corpus_rebuild_redownloads_pdf_with_mismatched_sha256_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """R2: `fetch_corpus`'s rebuild pass must verify `sha256` per file, not accept a
    PDF already on disk unconditionally. A correct-hash file is skipped (no network
    request); a mismatched-hash file is re-downloaded and replaced."""
    import hashlib

    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"
    pdf_dir.mkdir(parents=True)
    store = get_record_store("jsonl")

    from docsense.ingest.io import append_manifest
    from docsense.ingest.schema import ManifestRecord

    # correct hash on disk: must be skipped, no network request for it
    good_body = b"%PDF-1.4 already here, correct hash"
    (pdf_dir / "2301.00062v1.pdf").write_bytes(good_body)
    good_record = ManifestRecord(
        arxiv_id="2301.00062",
        version=1,
        title="Good On Disk",
        categories=["cs.CL"],
        query_category="cs.CL",
        published="2023-01-10T09:00:00Z",
        pdf_path="data/raw/2301.00062v1.pdf",
        sha256=hashlib.sha256(good_body).hexdigest(),
        num_bytes=len(good_body),
        fetched_at="2023-01-01T00:00:00+00:00",
    )

    # stale/corrupted hash on disk: must be re-downloaded and replaced
    stale_body = b"%PDF-1.4 stale, corrupted on disk"
    (pdf_dir / "2301.00063v1.pdf").write_bytes(stale_body)
    fresh_body = b"%PDF-1.4 freshly re-downloaded"
    mismatched_record = ManifestRecord(
        arxiv_id="2301.00063",
        version=1,
        title="Stale On Disk",
        categories=["cs.CL"],
        query_category="cs.CL",
        published="2023-01-10T09:00:00Z",
        pdf_path="data/raw/2301.00063v1.pdf",
        sha256=hashlib.sha256(fresh_body).hexdigest(),  # what re-download produces
        num_bytes=len(fresh_body),
        fetched_at="2023-01-01T00:00:00+00:00",
    )
    append_manifest(good_record, manifest_path, store)
    append_manifest(mismatched_record, manifest_path, store)

    def _fake_http_get(url: str, timeout_s: float) -> bytes:
        if url == "https://arxiv.org/pdf/2301.00062v1":
            raise AssertionError("correct-hash PDF on disk must not trigger a request")
        if url == "https://arxiv.org/pdf/2301.00063v1":
            return fresh_body
        raise AssertionError(f"unexpected request: {url}")

    monkeypatch.setattr(fetch_arxiv, "_http_get", _fake_http_get)

    config = _base_fetch_config(categories=["cs.CL"], max_papers=0)
    report = fetch_corpus(config, manifest_path, pdf_dir)

    assert not report.failed  # fresh re-download's sha256 matched the manifest
    assert (pdf_dir / "2301.00062v1.pdf").read_bytes() == good_body  # untouched
    assert (pdf_dir / "2301.00063v1.pdf").read_bytes() == fresh_body  # replaced


# --- R4: dry-run discovery must match what fetch_corpus actually appends ----


def test_dry_run_candidates_match_real_fetch_corpus_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`iter_new_candidates` (consumed by both the dry-run script and `fetch_corpus`)
    must report exactly the ids a subsequent real run appends, even starting from a
    partially-filled manifest."""
    from docsense.ingest.io import append_manifest
    from docsense.ingest.schema import ManifestRecord

    categories = ["cs.CL", "cs.LG"]
    cl_entries = [_entry("2301.00080", ["cs.CL"]), _entry("2301.00081", ["cs.CL"])]
    lg_entries = [_entry("2301.00082", ["cs.LG"]), _entry("2301.00083", ["cs.LG"])]
    feeds_by_category = {
        "cs.CL": [_make_feed(cl_entries)],
        "cs.LG": [_make_feed(lg_entries)],
    }
    all_entries = [*cl_entries, *lg_entries]
    pdf_bodies = {e["pdf_url"]: b"%PDF-1.4 body " + e["arxiv_id"].encode() for e in all_entries}

    transport = FakeArxivTransport(feeds_by_category=feeds_by_category, pdf_bodies=pdf_bodies)
    monkeypatch.setattr(fetch_arxiv, "_http_get", transport)

    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"
    store = get_record_store("jsonl")

    # pre-populate the manifest with one cs.CL paper already fetched
    import hashlib

    existing_body = b"%PDF-1.4 x"
    pre_existing = ManifestRecord(
        arxiv_id="2301.00080",
        version=1,
        title="Paper 2301.00080",
        categories=["cs.CL"],
        query_category="cs.CL",
        published="2023-01-10T09:00:00Z",
        pdf_path="data/raw/2301.00080v1.pdf",
        sha256=hashlib.sha256(existing_body).hexdigest(),
        num_bytes=len(existing_body),
        fetched_at="2023-01-01T00:00:00+00:00",
    )
    # its PDF already exists on disk with a matching sha256, so the rebuild/verify
    # pass (R2) is a no-op for it
    (pdf_dir / "2301.00080v1.pdf").parent.mkdir(parents=True)
    (pdf_dir / "2301.00080v1.pdf").write_bytes(existing_body)
    append_manifest(pre_existing, manifest_path, store)

    config = _base_fetch_config(categories=categories, max_papers=4)

    dry_run_candidates = {
        (category, entry["arxiv_id"])
        for category, entry in iter_new_candidates(config, manifest_path)
    }

    report = fetch_corpus(config, manifest_path, pdf_dir)
    appended = read_manifest(manifest_path, store)[1:]  # skip the pre-existing record
    real_candidates = {(record.query_category, record.arxiv_id) for record in appended}

    assert not report.failed
    assert dry_run_candidates == real_candidates
    assert dry_run_candidates == {
        ("cs.CL", "2301.00081"),
        ("cs.LG", "2301.00082"),
        ("cs.LG", "2301.00083"),
    }


# --- R19: rebuild pass must be visible to --dry-run --------------------------


def test_iter_rebuild_candidates_finds_only_missing_pdfs(tmp_path: Path):
    """`iter_rebuild_candidates` (factored out so `--dry-run` can see the rebuild
    pass, R19) yields exactly the manifest records whose PDF is absent from
    `pdf_dir`, leaving ones already on disk with a matching `sha256` alone."""
    import hashlib

    from docsense.ingest.io import append_manifest
    from docsense.ingest.schema import ManifestRecord

    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"
    pdf_dir.mkdir()
    store = get_record_store("jsonl")

    present_body = b"%PDF-1.4 x"
    present = ManifestRecord(
        arxiv_id="2301.00090",
        version=1,
        title="Present",
        categories=["cs.CL"],
        query_category="cs.CL",
        published="2023-01-10T09:00:00Z",
        pdf_path="data/raw/2301.00090v1.pdf",
        sha256=hashlib.sha256(present_body).hexdigest(),
        num_bytes=len(present_body),
        fetched_at="2023-01-01T00:00:00+00:00",
    )
    missing = ManifestRecord(
        arxiv_id="2301.00091",
        version=1,
        title="Missing",
        categories=["cs.CL"],
        query_category="cs.CL",
        published="2023-01-10T09:00:00Z",
        pdf_path="data/raw/2301.00091v1.pdf",
        sha256="b" * 64,
        num_bytes=1,
        fetched_at="2023-01-01T00:00:00+00:00",
    )
    append_manifest(present, manifest_path, store)
    append_manifest(missing, manifest_path, store)
    (pdf_dir / "2301.00090v1.pdf").write_bytes(present_body)

    config = _base_fetch_config(categories=["cs.CL"], max_papers=0)
    rebuild_ids = [r.arxiv_id for r in iter_rebuild_candidates(config, manifest_path, pdf_dir)]

    assert rebuild_ids == ["2301.00091"]


def test_iter_rebuild_candidates_flags_sha256_mismatch_on_disk(tmp_path: Path):
    """R2: a PDF already present on disk is still a rebuild candidate if its
    `sha256` no longer matches the manifest -- a stale/corrupted file is never
    trusted unconditionally just because it exists."""
    import hashlib

    from docsense.ingest.io import append_manifest
    from docsense.ingest.schema import ManifestRecord

    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"
    pdf_dir.mkdir()
    store = get_record_store("jsonl")

    on_disk_body = b"%PDF-1.4 corrupted or stale bytes"
    record = ManifestRecord(
        arxiv_id="2301.00095",
        version=1,
        title="Stale",
        categories=["cs.CL"],
        query_category="cs.CL",
        published="2023-01-10T09:00:00Z",
        pdf_path="data/raw/2301.00095v1.pdf",
        sha256=hashlib.sha256(b"%PDF-1.4 the real, expected bytes").hexdigest(),
        num_bytes=1,
        fetched_at="2023-01-01T00:00:00+00:00",
    )
    append_manifest(record, manifest_path, store)
    (pdf_dir / "2301.00095v1.pdf").write_bytes(on_disk_body)

    config = _base_fetch_config(categories=["cs.CL"], max_papers=0)
    rebuild_ids = [r.arxiv_id for r in iter_rebuild_candidates(config, manifest_path, pdf_dir)]

    assert rebuild_ids == ["2301.00095"]
    # read-only: the stale file is left in place for fetch_corpus to overwrite
    assert (pdf_dir / "2301.00095v1.pdf").read_bytes() == on_disk_body


# --- R20: a permanently-failing candidate must not block later candidates ---


def test_fetch_corpus_permanent_failure_does_not_block_later_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A permanently-failing candidate (e.g. arXiv serving a non-PDF body for a
    withdrawn paper) must not consume a quota slot: the category should still reach
    its target by moving on to a later, valid candidate in the same feed page, and
    the failure must still be reported (R20)."""
    entries = [
        _entry("2301.00020", ["cs.CL"]),  # permanently fails: never a valid PDF
        _entry("2301.00021", ["cs.CL"]),
        _entry("2301.00022", ["cs.CL"]),
    ]
    feed = _make_feed(entries)
    pdf_bodies: dict[str, bytes | Exception] = {
        entries[0]["pdf_url"]: b"this is not a pdf body at all",
        entries[1]["pdf_url"]: b"%PDF-1.4 ok body 1",
        entries[2]["pdf_url"]: b"%PDF-1.4 ok body 2",
    }
    transport = FakeArxivTransport(feeds_by_category={"cs.CL": [feed]}, pdf_bodies=pdf_bodies)
    monkeypatch.setattr(fetch_arxiv, "_http_get", transport)

    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"
    config = _base_fetch_config(max_papers=2)

    report = fetch_corpus(config, manifest_path, pdf_dir)

    # The category still reaches its target of 2 by moving past the permanently
    # failing candidate to the next valid ones in the same feed page.
    assert report.newly_fetched == 2
    assert report.newly_fetched_by_category["cs.CL"] == 2
    assert len(report.failed) == 1
    assert report.failed[0]["arxiv_id"] == "2301.00020"

    store = get_record_store("jsonl")
    records = read_manifest(manifest_path, store)
    assert {r.arxiv_id for r in records} == {"2301.00021", "2301.00022"}


def test_fetch_corpus_transient_failure_self_heals_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Pins the existing self-healing behavior (Tradeoffs #9) that must survive the
    R20 fix: a transient failure on run 1 is retried and succeeds on run 2, because
    the manifest never recorded it, so it is offered again."""
    entries = [_entry("2301.00030", ["cs.CL"]), _entry("2301.00031", ["cs.CL"])]
    feed = _make_feed(entries)
    pdf_bodies = {
        entries[0]["pdf_url"]: b"%PDF-1.4 ok on retry",
        entries[1]["pdf_url"]: b"%PDF-1.4 ok body " + entries[1]["arxiv_id"].encode(),
    }
    transport = FakeArxivTransport(feeds_by_category={"cs.CL": [feed]}, pdf_bodies=pdf_bodies)

    attempts = {"count": 0}

    def _flaky_then_ok(url: str, timeout_s: float) -> bytes:
        if url == entries[0]["pdf_url"]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise urllib.error.URLError("transient network blip")
        return transport(url, timeout_s)

    monkeypatch.setattr(fetch_arxiv, "_http_get", _flaky_then_ok)

    manifest_path = tmp_path / "manifest.jsonl"
    pdf_dir = tmp_path / "raw"
    config = _base_fetch_config(max_papers=2)

    first_report = fetch_corpus(config, manifest_path, pdf_dir)
    assert first_report.newly_fetched == 1
    assert len(first_report.failed) == 1
    assert first_report.failed[0]["arxiv_id"] == "2301.00030"

    second_report = fetch_corpus(config, manifest_path, pdf_dir)
    assert second_report.newly_fetched == 1
    assert not second_report.failed

    store = get_record_store("jsonl")
    records = read_manifest(manifest_path, store)
    assert {r.arxiv_id for r in records} == {"2301.00030", "2301.00031"}
