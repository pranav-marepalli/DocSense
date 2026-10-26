"""Resumable, category-stratified arXiv corpus fetcher.

Stdlib `urllib.request` + `xml.etree` (Tradeoffs #6): the arXiv API is one GET
returning Atom XML, and controlling pagination/rate-limiting explicitly (rather
than hiding it behind the `arxiv` package) is what makes a fetch resumable.
No API key needed.
"""

import hashlib
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Generator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docsense.ingest.io import RecordStore, append_manifest, get_record_store, read_manifest
from docsense.ingest.schema import FetchReport, ManifestRecord

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"

# src/docsense/ingest/fetch_arxiv.py -> repo root (same pattern as docsense.config.CONFIG_DIR).
REPO_ROOT = Path(__file__).resolve().parents[3]

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_ENTRY_ID_RE = re.compile(r"abs/([^v]+)v(\d+)")


class RateLimiter:
    """Blocks so consecutive `wait()` calls are >= `min_interval_s` apart."""

    def __init__(self, min_interval_s: float) -> None:
        """Create a limiter enforcing at least `min_interval_s` seconds between `wait()` calls."""
        self.min_interval_s = min_interval_s
        self._last_call: float | None = None

    def wait(self) -> None:
        """Sleep just long enough that this call is >= `min_interval_s` after the previous one."""
        now = time.monotonic()
        if self._last_call is not None:
            remaining = self.min_interval_s - (now - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


def _http_get(url: str, timeout_s: float) -> bytes:
    """The single network entry point in this module -- the only function tests must monkeypatch."""
    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310
        return response.read()


def build_search_query(category: str, published_after: str, published_before: str | None) -> str:
    """One stratum's query: `cat:<category> AND submittedDate:[<after>0000 TO <before>2359]`.

    `published_before=None` means today. Takes a single category by design --
    `fetch_corpus` issues one query per category (Tradeoffs #9).

    Raises:
        ValueError: on an empty category or a malformed date.
    """
    if not category:
        raise ValueError("category must not be empty")

    after = datetime.strptime(published_after, "%Y-%m-%d")
    if published_before is None:
        before_str = datetime.now(UTC).strftime("%Y%m%d")
    else:
        before_str = datetime.strptime(published_before, "%Y-%m-%d").strftime("%Y%m%d")

    after_str = after.strftime("%Y%m%d")
    return f"cat:{category} AND submittedDate:[{after_str}0000 TO {before_str}2359]"


def category_targets(categories: list[str], max_papers: int) -> dict[str, int]:
    """Even split: `max_papers // n` each, with the `max_papers % n` remainder going one-each to
    the first categories in config order, so targets sum to `max_papers`.

    Raises:
        ValueError: on an empty `categories`.
    """
    if not categories:
        raise ValueError("categories must not be empty")

    n = len(categories)
    base, remainder = divmod(max_papers, n)
    return {category: base + (1 if i < remainder else 0) for i, category in enumerate(categories)}


def fetch_result_page(
    search_query: str,
    start: int,
    page_size: int,
    timeout_s: float,
    limiter: RateLimiter,
    max_retries: int,
) -> str:
    """One GET, `sortBy=submittedDate&sortOrder=ascending` (stable paging, so a resumed run sees
    the same order). `limiter.wait()` before every attempt; retries `max_retries` then raises
    `urllib.error.URLError`. Returns raw Atom XML.
    """
    params = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
    )
    url = f"{ARXIV_API_URL}?{params}"

    last_error: Exception | None = None
    for _attempt in range(max_retries):
        limiter.wait()
        try:
            return _http_get(url, timeout_s).decode("utf-8")
        except urllib.error.URLError as exc:
            last_error = exc
            logger.warning("request to %s failed: %s", url, exc)

    raise urllib.error.URLError(f"exhausted {max_retries} retries for {url}") from last_error


def parse_atom_feed(xml_text: str) -> list[dict[str, Any]]:
    """One dict per entry: `arxiv_id`, `version`, `title` (whitespace-normalized), `categories`
    (primary first), `published`, `pdf_url`. Returns `[]` for an entry-less feed (the pagination
    stop signal).
    """
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        entry_id = entry.findtext(f"{_ATOM_NS}id") or ""
        match = _ENTRY_ID_RE.search(entry_id)
        if not match:
            continue
        arxiv_id, version = match.group(1), int(match.group(2))

        title = " ".join((entry.findtext(f"{_ATOM_NS}title") or "").split())
        published = entry.findtext(f"{_ATOM_NS}published") or ""

        categories: list[str] = []
        primary = entry.find(f"{_ARXIV_NS}primary_category")
        if primary is not None and primary.get("term"):
            categories.append(primary.get("term"))  # type: ignore[arg-type]
        for category_el in entry.findall(f"{_ATOM_NS}category"):
            term = category_el.get("term")
            if term and term not in categories:
                categories.append(term)

        pdf_url = None
        for link in entry.findall(f"{_ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break

        entries.append(
            {
                "arxiv_id": arxiv_id,
                "version": version,
                "title": title,
                "categories": categories,
                "published": published,
                "pdf_url": pdf_url,
            }
        )
    return entries


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def download_pdf(
    pdf_url: str, dest: Path, timeout_s: float, limiter: RateLimiter
) -> tuple[str, int]:
    """Download `pdf_url` to `dest.with_suffix('.part')`, then atomically rename to `dest`.

    A partial download never looks complete; the `.part` is removed on failure.

    Returns:
        `(sha256, num_bytes)` of the downloaded file.

    Raises:
        ValueError: if the response body lacks the `%PDF-` magic bytes.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(".part")
    limiter.wait()
    try:
        body = _http_get(pdf_url, timeout_s)
        if not body.startswith(b"%PDF-"):
            raise ValueError(f"response from {pdf_url} is not a PDF (missing %PDF- magic bytes)")
        part_path.write_bytes(body)
        part_path.rename(dest)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise

    return sha256_file(dest), dest.stat().st_size


def iter_new_candidates(
    config: dict,
    manifest_path: Path,
    limiter: RateLimiter | None = None,
    on_request: Callable[[], None] | None = None,
) -> Generator[tuple[str, dict[str, Any]], bool | None, None]:
    """Shared per-category discovery loop (decision 4; extracted so `fetch_corpus` and
    `scripts/fetch_arxiv.py --dry-run` cannot drift, R4).

    For each category in `fetch.categories`, pages that category's query until it has
    *confirmed* that category's `category_targets()` share of new candidates -- counted
    by `query_category` against the existing manifest, so a resumed run refills the
    right strata -- or the feed is exhausted. Skips any `arxiv_id` already in the
    manifest or already yielded by this call, so a cross-listed paper is offered to
    exactly one stratum. Performs no downloads.

    Args:
        limiter: shared `RateLimiter` for the Atom API requests this makes; a fresh
            one is created if not given (so a standalone caller like `--dry-run`
            still rate-limits itself).
        on_request: called once per Atom API page fetched, so a caller that also
            makes other requests (e.g. `fetch_corpus`'s downloads) can keep a single
            `api_requests` count.

    Send (R20):
        A caller driving this generator manually -- `gen.send(confirmed)` instead of
        a bare `next()`/`for` -- tells it whether the candidate it just yielded was
        confirmed downloaded (`True`) or permanently failed (`False`). Only a
        confirmed acceptance consumes the category's quota slot, so one
        permanently-failing candidate no longer blocks later, valid candidates in the
        same feed from reaching the target. `fetch_corpus` drives it this way. A plain
        `for`-loop consumer (e.g. `scripts/fetch_arxiv.py --dry-run`) never sends a
        value, so `confirmed` is `None`; that is treated the same as `True`
        (offer-based counting), which is the right estimate for a dry run that never
        downloads anything and so cannot know what would succeed.

    Yields:
        `(query_category, entry)` pairs, `entry` being one `parse_atom_feed` dict.
    """
    fetch_config = config["fetch"]
    store = get_record_store(config["record_format"])
    existing_records = read_manifest(manifest_path, store)
    seen_ids = {record.arxiv_id for record in existing_records}
    counts_by_category = Counter(record.query_category for record in existing_records)

    categories: list[str] = fetch_config["categories"]
    targets = category_targets(categories, fetch_config["max_papers"])
    if limiter is None:
        limiter = RateLimiter(fetch_config["request_delay_seconds"])

    for category in categories:
        query = build_search_query(
            category, fetch_config["published_after"], fetch_config["published_before"]
        )
        target = targets[category]
        have = counts_by_category.get(category, 0)
        start = 0

        while have < target:
            xml_text = fetch_result_page(
                query,
                start,
                fetch_config["page_size"],
                fetch_config["timeout_seconds"],
                limiter,
                fetch_config["max_retries"],
            )
            if on_request is not None:
                on_request()
            entries = parse_atom_feed(xml_text)
            if not entries:
                break

            for entry in entries:
                if have >= target:
                    break
                arxiv_id = entry["arxiv_id"]
                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)
                confirmed = yield category, entry
                if confirmed is None or confirmed:
                    have += 1

            start += fetch_config["page_size"]


def iter_rebuild_candidates(
    config: dict, manifest_path: Path, pdf_dir: Path
) -> Iterator[ManifestRecord]:
    """Manifest records that need (re-)downloading to match the manifest's `sha256`.

    Factored out of `fetch_corpus`'s rebuild pass (decision 2) the same way discovery
    was factored into `iter_new_candidates` (R4), so `fetch_corpus`'s rebuild pass and
    `scripts/fetch_arxiv.py --dry-run` share one predicate for "what the
    rebuild-from-manifest flow would re-download" and cannot drift (R19). This is the
    case a fresh clone hits: a full manifest, no PDFs on disk yet. It also catches a
    stale/corrupted on-disk PDF (R2): a file already present is still a candidate if
    its `sha256` no longer matches the manifest's recorded value, logged as a
    warning -- "verifies sha256 per file" (design §3) applies to files already on
    disk, not just missing ones.

    Read-only: performs a manifest read, `Path.exists()` checks, and a `sha256_file`
    hash of any PDF already present, but never deletes or otherwise mutates
    `pdf_dir` -- the actual re-download (and the stale file being atomically replaced
    by it) happens in `fetch_corpus`'s rebuild loop, so this stays safe for
    `--dry-run` to call.

    Yields:
        `ManifestRecord`s, in manifest order, whose recorded `pdf_path` is either
        absent from `pdf_dir` or present with a `sha256` mismatch.
    """
    store = get_record_store(config["record_format"])
    for record in read_manifest(manifest_path, store):
        dest = pdf_dir / f"{record.arxiv_id}v{record.version}.pdf"
        if not dest.exists():
            yield record
            continue
        actual_sha256 = sha256_file(dest)
        if actual_sha256 != record.sha256:
            logger.warning(
                "sha256 mismatch for %s already on disk (expected %s, got %s); "
                "will be re-downloaded",
                record.arxiv_id,
                record.sha256,
                actual_sha256,
            )
            yield record


def fetch_corpus(config: dict, manifest_path: Path, pdf_dir: Path) -> FetchReport:
    """Resumable, category-stratified fetch driver (decision 4).

    Two passes. First, decision 2's rebuild path: `iter_rebuild_candidates` finds any
    existing manifest record whose PDF is missing on disk (e.g. a fresh clone) *or*
    present but failing `sha256` verification against the manifest (R2 -- a stale or
    corrupted file on disk is never trusted unconditionally), and each is
    re-downloaded at its recorded `version`; a `sha256` mismatch on the freshly
    downloaded bytes is itself a per-paper failure in `FetchReport.failed`, never a
    silent substitution, and the mismatched file is removed. Second,
    `iter_new_candidates` discovers new papers and
    this function downloads them up to each category's `category_targets()` share,
    driving the generator with `.send()` so only a *confirmed* download consumes a
    quota slot (R20) -- a permanently-failing candidate cannot block later, valid
    candidates in the same feed from reaching the target. Skips any `arxiv_id` already
    in the manifest, so a cross-listed paper downloads once and counts toward the
    first stratum that saw it. Appends each `ManifestRecord` immediately (an
    interrupted run loses at most one paper); per-paper errors go to
    `FetchReport.failed`, never abort. A thin category leaves the corpus under
    `max_papers`: accepted, no backfill (Tradeoffs #9).
    """
    fetch_config = config["fetch"]
    started_at = datetime.now(UTC).isoformat()

    store: RecordStore = get_record_store(config["record_format"])
    existing_records = read_manifest(manifest_path, store)

    categories: list[str] = fetch_config["categories"]
    targets = category_targets(categories, fetch_config["max_papers"])
    limiter = RateLimiter(fetch_config["request_delay_seconds"])

    pdf_dir.mkdir(parents=True, exist_ok=True)

    search_queries = {
        category: build_search_query(
            category, fetch_config["published_after"], fetch_config["published_before"]
        )
        for category in categories
    }
    newly_fetched_by_category: dict[str, int] = dict.fromkeys(categories, 0)
    failed: list[dict[str, str]] = []
    api_requests = 0

    def _count_request() -> None:
        nonlocal api_requests
        api_requests += 1

    for record in iter_rebuild_candidates(config, manifest_path, pdf_dir):
        dest = pdf_dir / f"{record.arxiv_id}v{record.version}.pdf"
        pdf_url = f"https://arxiv.org/pdf/{record.arxiv_id}v{record.version}"
        try:
            sha256, _num_bytes = download_pdf(
                pdf_url, dest, fetch_config["timeout_seconds"], limiter
            )
        except Exception as exc:
            failed.append({"arxiv_id": record.arxiv_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if sha256 != record.sha256:
            dest.unlink(missing_ok=True)
            failed.append(
                {
                    "arxiv_id": record.arxiv_id,
                    "error": (
                        f"sha256 mismatch on rebuild: expected {record.sha256}, got {sha256}"
                    ),
                }
            )

    # Drive the discovery generator manually (rather than a bare `for`) so each
    # candidate's download outcome is sent back via `.send()` (R20): only a
    # confirmed download consumes that category's quota slot, so one
    # permanently-failing candidate does not block later, valid candidates in the
    # same feed from reaching the target.
    newly_fetched = 0
    candidate_gen = iter_new_candidates(
        config, manifest_path, limiter=limiter, on_request=_count_request
    )
    confirmed_last: bool | None = None
    while True:
        try:
            if confirmed_last is None:
                category, entry = next(candidate_gen)
            else:
                category, entry = candidate_gen.send(confirmed_last)
        except StopIteration:
            break

        arxiv_id = entry["arxiv_id"]
        dest = pdf_dir / f"{arxiv_id}v{entry['version']}.pdf"
        try:
            sha256, num_bytes = download_pdf(
                entry["pdf_url"], dest, fetch_config["timeout_seconds"], limiter
            )
        except Exception as exc:
            failed.append({"arxiv_id": arxiv_id, "error": f"{type(exc).__name__}: {exc}"})
            confirmed_last = False
            continue

        record = ManifestRecord(
            arxiv_id=arxiv_id,
            version=entry["version"],
            title=entry["title"],
            categories=entry["categories"],
            query_category=category,
            published=entry["published"],
            pdf_path=os.path.relpath(dest, REPO_ROOT),
            sha256=sha256,
            num_bytes=num_bytes,
            fetched_at=datetime.now(UTC).isoformat(),
        )
        append_manifest(record, manifest_path, store)
        newly_fetched += 1
        newly_fetched_by_category[category] += 1
        confirmed_last = True

    return FetchReport(
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        search_queries=search_queries,
        max_papers=fetch_config["max_papers"],
        category_targets=targets,
        already_in_manifest=len(existing_records),
        newly_fetched=newly_fetched,
        newly_fetched_by_category=newly_fetched_by_category,
        failed=failed,
        api_requests=api_requests,
    )

