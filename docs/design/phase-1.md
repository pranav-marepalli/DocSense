# Phase 1: Ingestion

## 1. Goal

Build the corpus and turn it into structured, page-aware, chunked records on disk that Phase 2
(indexing) and later phases consume without touching a PDF again. Two halves: (a) a resumable arXiv
**fetch** script that queries the API once per category over a date range and writes a committed
`data/manifest.jsonl`; (b) **extraction** producing per-page text, tables as markdown, figure crops
with captions, and text chunks that each carry the page number(s) they came from — the page-level
citation contract the project depends on. Measured by: every acceptance criterion in §6 passing
(`check` green); a fixture-corpus end-to-end run matching the §3 schemas; and an ingestion report
whose aggregates match the per-paper records. Extraction *quality* is not graded against a gold set
here — that surfaces indirectly in Phase 3 eval.

## 2. Module plan

New files under `src/docsense/ingest/` (currently only an empty `__init__.py`), two CLI scripts, one
`.gitignore` change.

### `src/docsense/ingest/schema.py` — pydantic models, no I/O

The two models the whole pipeline is built around:

```python
class Page(BaseModel):
    """One page. page_number is 1-indexed and is the citation unit everywhere downstream."""
    page_number: int
    text: str
    blocks: list[Block]     # Block: bbox tuple[float,float,float,float] (PDF points) + text str
    char_count: int
    low_text_layer: bool    # char_count < min_text_chars_per_page; page is still kept

class Chunk(BaseModel):
    """A retrieval-sized passage. The record Phase 2 indexes."""
    chunk_id: str           # f"{arxiv_id}_chunk_{i:04d}"
    arxiv_id: str
    chunk_index: int        # position within the paper, 0-indexed
    page_start: int
    page_end: int
    text: str
    token_count: int        # whitespace approximation, Tradeoffs #3
    source: Literal["text"] # "table"/"figure_caption" land here in Phase 5
```

The rest are flat `BaseModel`s; fields in order, all required unless typed `| None`:

- **`Table`** — detected table; `raw_text` always populated, `markdown` only when detection
  succeeded: `table_id: str` (`f"{arxiv_id}_table_{i:03d}"`), `page_number: int`,
  `bbox: tuple[float, float, float, float]`, `caption: str | None`, `markdown: str | None`,
  `raw_text: str`, `method: Literal["find_tables", "raw_text_fallback"]`.
- **`Figure`** — crop + nearby caption; finalized contract for Phase 5: `figure_id: str`
  (`f"{arxiv_id}_figure_{i:03d}"`), `page_number: int`, `bbox: tuple[float, float, float, float]`,
  `caption: str | None`, `image_path: str` (cropped PNG, relative to `out_dir`).
- **`PaperDocument`** — full per-paper record at `data/parsed/papers/<arxiv_id>.json`:
  `arxiv_id: str`, `source_pdf: str` (relative to repo root), `title: str | None`, `num_pages: int`,
  `extracted_at: str` (ISO 8601 UTC), `extractor_version: str` (e.g. `"pymupdf-1.24.10"`),
  `pages: list[Page]`, `tables: list[Table]`, `figures: list[Figure]`, `chunks: list[Chunk]`.
- **`ManifestRecord`** — one fetched paper, committed to `data/manifest.jsonl`; the corpus is
  rebuildable from these alone: `arxiv_id: str` (bare, e.g. `"2301.12345"`), `version: int` (the
  version actually downloaded), `title: str`, `categories: list[str]` (primary first),
  `query_category: str` (the stratum whose query returned it — not derivable from `categories` for
  cross-listed papers, and what makes the final split auditable), `published: str` (v1 submission
  date), `pdf_path: str` (relative to repo root), `sha256: str`, `num_bytes: int`, `fetched_at: str`.
- **`PaperStatus`** — one row of the ingestion report: `arxiv_id: str`,
  `status: Literal["ok", "error"]`, `error: str | None` (exception type + message when
  `status == "error"`), `num_pages: int`, `num_chunks: int`, `num_tables: int`, `num_figures: int`,
  `num_low_text_pages: int`.
- **`IngestReport`** — one ingestion run, at `data/parsed/ingest_report.json`: `started_at: str`,
  `finished_at: str`, `extractor_version: str`, `papers_total: int`, `papers_ok: int`,
  `papers_failed: int`, `total_pages: int`, `total_chunks: int`, `tables_total: int`,
  `tables_by_method: dict[str, int]` (keys `"find_tables"`, `"raw_text_fallback"`),
  `figures_total: int`, `low_text_layer_pages: int`, `papers: list[PaperStatus]`.
- **`FetchReport`** — one fetch run, printed by the CLI, not persisted: `started_at: str`,
  `finished_at: str`, `search_queries: dict[str, str]` (category -> query issued), `max_papers: int`,
  `category_targets: dict[str, int]`, `already_in_manifest: int`, `newly_fetched: int`,
  `newly_fetched_by_category: dict[str, int]` (how close each stratum came to target),
  `failed: list[dict[str, str]]` (`[{"arxiv_id": ..., "error": ...}]`), `api_requests: int`.

### `src/docsense/ingest/fetch_arxiv.py` (new, decision 1)

Stdlib `urllib.request` + `xml.etree` (Tradeoffs #6). No API key needed.

```python
ARXIV_API_URL = "https://export.arxiv.org/api/query"

class RateLimiter:
    """Blocks so consecutive wait() calls are >= min_interval_s apart (3.0s from config)."""
    def __init__(self, min_interval_s: float) -> None: ...
    def wait(self) -> None: ...

def build_search_query(category: str, published_after: str,
                       published_before: str | None) -> str:
    """One stratum's query: cat:<category> AND submittedDate:[<after>0000 TO <before>2359];
    published_before=None means today. Takes a single category by design — fetch_corpus issues one
    query per category (Tradeoffs #9). ValueError on an empty category or a bad date."""
def category_targets(categories: list[str], max_papers: int) -> dict[str, int]:
    """Even split: max_papers // n each, the max_papers % n remainder giving one extra to the first
    categories in config order, so targets sum to max_papers. ValueError on empty categories."""
def fetch_result_page(search_query: str, start: int, page_size: int, timeout_s: float,
                      limiter: RateLimiter, max_retries: int) -> str:
    """One GET, sortBy=submittedDate&sortOrder=ascending (stable paging, so a resumed run sees
    the same order). limiter.wait() before every attempt; retries max_retries then raises
    urllib.error.URLError. Returns raw Atom XML."""
def parse_atom_feed(xml_text: str) -> list[dict[str, Any]]:
    """One dict per entry: arxiv_id, version, title (whitespace-normalized), categories,
    published, pdf_url. Returns [] for an entry-less feed (the pagination stop signal)."""
def sha256_file(path: Path) -> str:
    """Streaming SHA-256 hex digest of a file."""
def download_pdf(pdf_url: str, dest: Path, timeout_s: float,
                 limiter: RateLimiter) -> tuple[str, int]:
    """Downloads to dest.with_suffix('.part') then atomically renames, so a partial download never
    looks complete; removes the .part on failure. Returns (sha256, num_bytes); ValueError if the
    body lacks %PDF- magic bytes."""
def fetch_corpus(config: dict, manifest_path: Path, pdf_dir: Path) -> FetchReport:
    """Resumable, category-stratified driver (decision 4). Reads existing records via
    read_manifest(), then for each category in fetch.categories builds its own query and pages that
    feed until the manifest holds that category's category_targets() share — counted by
    query_category, so a resumed run refills the right strata — or the feed is exhausted. Skips any
    arxiv_id already in the manifest, so a cross-listed paper downloads once and counts toward the
    first stratum that saw it. Appends each ManifestRecord immediately (an interrupted run loses at
    most one paper); per-paper errors go to FetchReport.failed, never abort. A thin category leaves
    the corpus under max_papers: accepted, no backfill (Tradeoffs #9)."""
```

### `src/docsense/ingest/pdf_extract.py` — PyMuPDF; each takes an open `fitz.Document`

```python
def extract_title(doc: fitz.Document) -> str | None:
    """Metadata title if non-empty, else largest-font span on page 1, else None."""
def extract_pages(doc: fitz.Document, min_text_chars_per_page: int) -> list[Page]:
    """One Page per PDF page via page.get_text("blocks"), page_number from 1. Sets low_text_layer
    when char_count < min_text_chars_per_page and logs a warning naming the page; such pages are
    kept, never dropped (no OCR — Tradeoffs #7)."""
def extract_tables(doc: fitz.Document, arxiv_id: str, caption_radius_pts: float) -> list[Table]:
    """page.find_tables() per page; per hit Table.to_markdown() with method="find_tables", else (on
    raise or empty) get_text(clip=bbox) with markdown=None and method="raw_text_fallback". raw_text
    always filled. Caption = nearest block matching /^Table\\s+\\d+/ within caption_radius_pts
    above or below the bbox."""
def extract_figures(doc: fitz.Document, arxiv_id: str, image_dir: Path, dpi: int,
                    caption_radius_pts: float) -> list[Figure]:
    """page.get_image_info(xrefs=True) to locate embedded rasters, crop via
    page.get_pixmap(clip=bbox, dpi=dpi), write PNG under image_dir/<arxiv_id>/. Caption lookup
    mirrors extract_tables with /^Fig(ure)?\\.?\\s+\\d+/."""
```

### `src/docsense/ingest/chunk.py` — operates on `list[Page]`, independent of PDFs

```python
def count_tokens(text: str) -> int:
    """Approximate token count via whitespace splitting (Tradeoffs #3)."""
def chunk_pages(pages: list[Page], arxiv_id: str, chunk_size: int, overlap: int,
                min_chunk_tokens: int) -> list[Chunk]:
    """Concatenates page text in order tracking page boundaries, then slides a chunk_size-token
    window with `overlap` tokens of overlap. page_start/page_end are the first/last page any token
    in the chunk came from. A trailing chunk below min_chunk_tokens is merged into the previous
    chunk. low_text_layer pages contribute what text they have. ValueError if overlap >=
    chunk_size."""
```

### `src/docsense/ingest/io.py`

All record collections go through one abstraction, so Parquet later is a new class plus a config
value rather than a refactor (decision 2).

```python
class RecordStore(Protocol):
    """Append-oriented store for flat dict records (chunks, manifest rows). The only persistence
    interface callers may use for record collections."""
    extension: str
    def append(self, records: Iterable[dict[str, Any]], path: Path) -> None: ...
    def read_all(self, path: Path) -> list[dict[str, Any]]: ...
    def count(self, path: Path) -> int: ...

class JsonlRecordStore:
    """RecordStore over newline-delimited JSON. Creates parent dirs on append; read_all on a
    missing path returns []."""

def get_record_store(fmt: str = "jsonl") -> RecordStore:
    """Factory keyed by config `record_format`; ValueError on unknown fmt (Tradeoffs #2)."""
def write_chunks(chunks: list[Chunk], path: Path, store: RecordStore) -> None:
    """Appends chunks via store — the corpus-wide file Phase 2 reads."""
def append_manifest(record: ManifestRecord, path: Path, store: RecordStore) -> None:
    """Appends one record. Append-only keeps fetch runs resumable."""
def write_paper(doc: PaperDocument, out_dir: Path) -> Path:
    """Writes out_dir/papers/<arxiv_id>.json (model_dump_json, indent=2). Nested single documents
    bypass RecordStore by design (Tradeoffs #2)."""
def write_ingest_report(report: IngestReport, out_dir: Path) -> Path:
    """Writes out_dir/ingest_report.json (model_dump_json, indent=2)."""
```

Three readers mirror the writers, each validating into models and raising `pydantic.ValidationError`
on malformed input: `read_chunks(path, store) -> list[Chunk]`, `read_manifest(path, store) ->
list[ManifestRecord]` (returns `[]` if the file is absent — what makes a first fetch run work), and
`read_paper(path: Path) -> PaperDocument`.

### `src/docsense/ingest/pipeline.py`

```python
def ingest_pdf(pdf_path: Path, arxiv_id: str, image_dir: Path, config: dict) -> PaperDocument:
    """Opens pdf_path with fitz, runs extract_title/pages/tables/figures and chunk_pages, returns
    a PaperDocument. Does not write to disk."""
def ingest_corpus(out_dir: Path, config: dict, manifest_path: Path | None = None,
                  pdf_dir: Path | None = None) -> IngestReport:
    """Input list = manifest pdf_paths when manifest_path is given (the canonical path), else glob
    pdf_dir/*.pdf. Per paper: ingest_pdf, write_paper, append chunks to out_dir/chunks.jsonl; on
    exception log it and record a PaperStatus with status="error", continuing when
    config["skip_on_error"] else re-raising. Finally writes and returns the IngestReport whose
    aggregates are summed from its PaperStatus rows."""
```

### Scripts and `.gitignore`

- `scripts/fetch_arxiv.py` (committed, see §3): `--config ingest [--max-papers N] [--dry-run]`;
  loads `config/ingest.yaml`, calls `fetch_corpus`, prints the `FetchReport` including per-category
  target vs. actual. `--dry-run` lists what would be downloaded, writing nothing. Non-zero exit if
  any download failed.
- `scripts/ingest.py`: `--config ingest [--manifest PATH | --pdf-dir DIR] [--out-dir DIR]`; calls
  `ingest_corpus`, prints the report, exits non-zero if any paper errored.
- `.gitignore`: replace the blanket `data/` entry with `data/raw/` and `data/parsed/` so
  `data/manifest.jsonl` (and Phase 3's eval set) are tracked.

## 3. Data flow

```
config/ingest.yaml -> scripts/fetch_arxiv.py -> arXiv API (one query per category, >=3s apart)
        +--> data/raw/<arxiv_id>v<version>.pdf     (gitignored)
        +--> data/manifest.jsonl                   (COMMITTED, append-only)
                |  ingest_pdf() per manifest record
        PaperDocument (in memory)
                +--> data/parsed/papers/<arxiv_id>.json
                +--> data/parsed/images/<arxiv_id>/figure_*.png
                +--> data/parsed/chunks.jsonl      (append, one Chunk per line)
                |  after the whole corpus
                +--> data/parsed/ingest_report.json
```

`data/parsed/chunks.jsonl` is the Phase 2 contract: indexing needs only `chunk_id`, `arxiv_id`,
`page_start`, `page_end`, `text`. The per-paper JSON is the richer record for Phase 5 and debugging.
**Formats:** `manifest.jsonl` / `chunks.jsonl` are newline-delimited JSON, one `ManifestRecord` /
`Chunk` per line, written only via `RecordStore`; `papers/<arxiv_id>.json` and `ingest_report.json`
are pretty-printed single objects matching `PaperDocument` / `IngestReport`. Manifest line order
follows fetch order, so it is grouped by `query_category`; nothing downstream may depend on that
order and Phase 3 sampling must shuffle with a seed.

**Licensing / what is committed** (decision 3). arXiv papers are not uniformly licensed for
redistribution, so `data/raw/` (PDFs) and `data/parsed/` (text, crops, chunks) stay gitignored.
Committed: `scripts/fetch_arxiv.py`, `config/ingest.yaml`, `data/manifest.jsonl`, and later the
Phase 3 eval set. `arxiv_id` + `version` + `sha256` make the corpus exactly reproducible. **The
README must document the rebuild path**: clone, `python scripts/fetch_arxiv.py --config ingest`
(re-fetches exactly the manifest's ids, verifying `sha256` per file), then `python scripts/ingest.py
--config ingest`. `arxiv_id` comes from the manifest on manifest-driven runs; for `--pdf-dir` runs
it is the filename stem, and if the stem doesn't match `\d{4}\.\d{4,5}` the raw stem is used and a
warning is logged.

## 4. Config keys (`config/ingest.yaml`)

Stratification has no toggle: it is the only fetch mode (§7, Tradeoffs #9).

| Key | Default | Meaning |
|---|---|---|
| `fetch.categories` | `[cs.CL, cs.LG, cs.CV, cs.IR]` | Categories; one query + one quota each |
| `fetch.published_after` | `"2023-01-01"` | Lower bound on submission date |
| `fetch.published_before` | `null` | Upper bound; `null` = today |
| `fetch.max_papers` | `300` | Target corpus size, split evenly across `categories` |
| `fetch.request_delay_seconds` | `3.0` | Min gap between any two arXiv requests |
| `fetch.page_size` | `100` | `max_results` per API call |
| `fetch.timeout_seconds` | `60` | Per-request socket timeout |
| `fetch.max_retries` | `3` | Retries per request, each still rate-limited |
| `manifest_path` | `"data/manifest.jsonl"` | Committed corpus manifest |
| `pdf_dir` | `"data/raw"` | PDFs (fetch writes, ingest reads) |
| `out_dir` | `"data/parsed"` | Extraction output root |
| `record_format` | `"jsonl"` | `RecordStore` backend |
| `chunk_size_tokens` | `300` | Target chunk size, whitespace tokens |
| `chunk_overlap_tokens` | `50` | Overlap between consecutive chunks |
| `min_chunk_tokens` | `50` | Trailing chunks below this are merged, not dropped |
| `min_text_chars_per_page` | `100` | Below this, page flagged `low_text_layer` |
| `table_detection` | `true` | Run `extract_tables` |
| `figure_extraction` | `true` | Run `extract_figures` |
| `figure_dpi` | `150` | Rasterization DPI for figure crops |
| `caption_search_radius_pts` | `40` | Vertical caption search window (PDF points) |
| `skip_on_error` | `true` | Continue the batch past a single bad PDF |

## 5. Tradeoffs

1. **PyMuPDF vs. Docling.** PyMuPDF: already pinned, no model weights, fast, and
   `find_tables()`/`get_image_info()` cover tables and figures with no extra library. Docling reads
   order and tables better but pulls in torch and is far slower per page. Extraction stays behind
   `pdf_extract.py`, so Docling could replace it later without touching `chunk.py` or the schema.
2. **JSONL now vs. Parquet now vs. JSONL behind an interface.** At ~300 papers JSONL is plainly
   sufficient: no `pyarrow`, diffable, and append-as-you-go is what makes fetch resumable. Rather
   than scatter `json.loads` across modules, record collections go through `RecordStore`, so Parquet
   later is one class plus `record_format: parquet`. Nested single documents stay plain JSON.
3. **Chunking: fixed-size window + whitespace tokens vs. semantic chunking + a real tokenizer.**
   Fixed-size wins on determinism and testability with no extra dependency, and ingestion avoids a
   tokenizer tied to one embedder since the embedder varies by ablation config (Phase 2 recomputes
   exact subword counts). Semantic chunking is a Phase 4/7 ablation.
4. **Captions: proximity heuristic vs. none.** Regex + vertical distance is best-effort and will
   mis-attach on some multi-column layouts. Accepted: captions matter for Phase 5, and
   `caption: None` is a safe, visible failure — nothing downstream may assume captions exist.
5. **Errors: skip-and-log vs. fail-fast.** Default `skip_on_error: true`; one malformed PDF must not
   abort a 300-paper run. `false` stays available for debugging a single file.
6. **arXiv client: stdlib vs. `arxiv` package vs. `feedparser`.** Stdlib `urllib` + `xml.etree`: the
   API is one GET returning Atom, and the `arxiv` package hides the rate limiting and pagination we
   must control explicitly for resumability. Zero new dependencies.
7. **Low-text pages: OCR vs. drop vs. flag-and-keep.** Flag and keep (decision 5). OCR is out of
   scope. Dropping would silently break page numbering and the citation contract; flagging keeps
   page numbers honest and surfaces the count in `ingest_report.json`.
8. **Tables: `find_tables()` markdown vs. pdfplumber/camelot.** PyMuPDF only (decision 6), with
   `raw_text_fallback` when detection fails and a `method` field so Phase 5/7 can measure how often
   the fallback fires. The earlier draft's `cells` is replaced by `markdown`, which is what an LLM
   consumes anyway; `raw_text` is present either way.
9. **Category coverage: one combined OR'd query vs. per-category stratified queries vs.
   volume-proportional quotas.** Stratified, decision 4: n queries instead of 1 (a few more
   requests, identical per-paper download cost) but cs.IR cannot be swamped by cs.LG volume, so
   Phase 3 eval balance is a property of the corpus rather than something to fix during sampling.
   The combined query was the earlier draft and skews to the highest-volume category; proportional
   quotas preserve that skew deliberately, wrong for a balanced eval set. Accepted cost: an
   under-`max_papers` corpus when a category is thin, with no backfill — backfilling would quietly
   reintroduce the skew it is meant to prevent.

**Eval leakage.** No risk in this phase: ingestion produces only corpus chunks — no training
examples, labels, or synthetic data. Two guardrails to carry forward: (a) the Phase 3 gold set must
be drawn from this corpus *after* the committed manifest freezes it, so retrieval cannot be tuned
against papers chosen later; (b) when Phase 4 generates synthetic training data from `chunks.jsonl`,
`eval-leakage-check` runs against chunk text — which is why `chunk_id` must be stable and
deterministic for a given config.

## 6. Acceptance criteria

1. `build_search_query("cs.CL", ...)` emits the expected string (one `cat:` term ANDed with a
   `submittedDate` range) and raises `ValueError` on an empty category and on a malformed date.
   `category_targets` splits 10 over 4 categories as `{3, 3, 2, 2}` (sum == `max_papers`, max
   pairwise difference 1) and raises `ValueError` on empty categories.
2. `parse_atom_feed` on a committed fixture Atom XML returns the expected `arxiv_id`, `version`,
   `title`, `categories`, `published`, `pdf_url`, and returns `[]` for an entry-less feed.
3. `RateLimiter` enforces the gap: with `min_interval_s=0.05`, three `wait()` calls take >= 0.1s
   (measured or monkeypatched clock — never a real 3s sleep).
4. `fetch_corpus` is resumable and offline-testable: with the request helper monkeypatched and a
   manifest already holding one of three returned ids, it downloads exactly the two new papers,
   appends exactly two `ManifestRecord`s, leaves the pre-existing line byte-identical, and a second
   run downloads nothing. A download raising mid-run leaves no `.pdf` (only a cleaned-up `.part`)
   and lands in `FetchReport.failed`.
5. `fetch_corpus` stratifies: with 4 categories and `max_papers: 8` it issues one query per category
   (4 entries in `FetchReport.search_queries`), appends at most that category's target per
   `query_category`, downloads a paper cross-listed into two queried categories exactly once, and
   when one category's feed is exhausted early returns a corpus under `max_papers` while the other
   three reach their targets (`newly_fetched_by_category` shows the shortfall).
6. `extract_pages` on an in-test `fitz` fixture (3 pages, known text) returns 3 `Page`s with correct
   1-indexed `page_number`s and matching text; a near-empty page is flagged `low_text_layer=True`
   and is still present in the returned list.
7. `chunk_pages` on synthetic `Page`s with known word counts produces the expected chunk count, with
   `chunk_overlap_tokens` of overlap verified by token comparison between consecutive chunks, raises
   `ValueError` when `overlap >= chunk_size`, and sets `page_start != page_end` matching the real
   source pages for a chunk forced across a page boundary.
8. `extract_tables`/`extract_figures` do not raise on: (a) a fixture PDF with one inserted image +
   one table-like text grid, detecting at least the image and producing a `Table` whose `method` is
   one of the two allowed values with non-empty `raw_text`; (b) a fixture PDF with neither,
   returning empty lists.
9. Every `schema.py` model round-trips `model_dump_json()` -> `model_validate_json()` to an equal
   object (schema fidelity test).
10. `JsonlRecordStore` satisfies `RecordStore`: append-then-`read_all` round-trips records in order,
    `read_all` on a missing path returns `[]`, `count` == `len(read_all)`, two appends accumulate
    rather than truncate, and `get_record_store("nope")` raises `ValueError`.
11. `ingest_corpus` on a 3-PDF fixture dir where one PDF is truncated still produces output for the
    other two and records the error in the report instead of raising, with `skip_on_error: true`.
12. **Ingestion summary report test:** after that run, `ingest_report.json` validates as
    `IngestReport` and its aggregates equal the sums over `papers`: `papers_ok`/`papers_failed`,
    `total_pages`, `total_chunks`, `tables_total == sum(tables_by_method.values())`,
    `figures_total`, `low_text_layer_pages` matching flagged pages across the written
    `PaperDocument`s, and the failed paper's `error` a non-empty string naming the cause.
13. `write_chunks` output has exactly one JSON object per line, and total lines after a corpus run
    equal the sum of `len(doc.chunks)` over successfully ingested papers.
14. `scripts/ingest.py` runs end-to-end on a fixture `pdf_dir`, exiting 0 when all files succeed and
    non-zero when any errors; `scripts/fetch_arxiv.py --dry-run` (network monkeypatched) writes
    nothing to `pdf_dir` or the manifest.
15. No test makes a real network request (enforced by monkeypatching the fetch module's single
    request helper), no test sleeps the real 3s delay, and `.gitignore` ignores `data/raw/` and
    `data/parsed/` but not `data/manifest.jsonl`.
16. `check` skill (pytest -q + ruff check .) passes with these tests added.

## 7. Decisions (formerly open questions, settled by the human)

Nothing here is open; the reviewer should treat all four as fixed.

- **Disk and runtime budget — accepted as-is.** ~3-10 GB for ~300 PDFs plus `figure_dpi: 150` crops,
  and a ~15-25 min fetch (>= 300 x 3s of deliberate delay) plus extraction, are acceptable on the
  target machine. `figure_dpi` and `fetch.max_papers` defaults stand; no design change.
- **Version pinning on rebuild — confirmed.** Fetch the version recorded in the manifest, not the
  latest, for strict reproducibility; a `sha256` mismatch is a per-paper failure in
  `FetchReport.failed`, never a silent substitution. No design change.
- **Category skew — resolved: stratify evenly.** `fetch_corpus` runs one `build_search_query` per
  category capped at its `category_targets()` share instead of one date-ordered query across all
  four (§2, Tradeoffs #9). Expected consequence: the corpus may land slightly under
  `fetch.max_papers` if a category is thin in the date range — acceptable, and reported rather than
  silently patched.
- **Figure crop storage — confirmed.** Crops regenerate on every rebuild; no cache directory inside
  or outside the repo. No design change.
