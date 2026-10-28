# Phase 3: Eval harness

## 1. Goal

Build the retrieval **evaluation harness** — Recall@k, MRR, nDCG@k — plus the on-disk gold-set
format and the human labeling scaffold that fills it. Phase 2 shipped retrieval with no way to say
whether it is any good; this phase makes "better" measurable so Phase 4's hybrid/reranker and Phase
7's ablation are comparisons rather than claims. Scope is retrieval only (Phase 2 decision 1
deferred generation), so no answer-quality metric exists yet. **The gold set ships empty.**
Implementation delivers the harness, the schema, the labeling CLI, and a committed zero-byte
`data/eval/gold_set.jsonl`; a human then labels it in the separate step specified in §7 — per
CLAUDE.md nothing in this repo may create, edit, or infer a label, and no code path produces label
content from a model, a heuristic, or a retrieval result. **Measured by** every §6 acceptance
criterion passing with `check` green, including metric values hand-computed in §6.1 and an
end-to-end `scripts/eval.py` run over both an empty and a tiny synthetic gold set. No real retrieval
numbers are produced or expected in this phase, and none may be quoted until the human labels.

## 2. Module plan

New files under `src/docsense/eval/` (currently an empty `__init__.py`), three CLI scripts, one
config, one committed data file plus its README, and two small changes to existing files.

### `src/docsense/eval/schema.py` — pydantic models, no I/O (the Phase 1/2 pattern)

```python
MAX_RELEVANCE = 2  # schema owns the grade scale; config only owns the binary threshold

class LabeledSpan(BaseModel):
    """One human relevance judgment, keyed on a PAGE SPAN rather than a chunk_id.

    chunk_ids encode the chunking config (Phase 1 `chunk_size_tokens`), so labels keyed on them
    would be invalidated by exactly the re-chunking ablations Phase 7 runs (Tradeoffs #1). A page
    span is chunking-invariant and is already the project's citation unit.
    """
    arxiv_id: str
    page_start: int          # 1-indexed, inclusive; <= page_end (validator)
    page_end: int
    relevance: int           # 0..MAX_RELEVANCE; 0 = judged NOT relevant (never a training negative, §5)
    note: str | None
    source_chunk_id: str | None  # provenance only: the chunk the labeler was shown. Never scored.

class GoldExample(BaseModel):
    """One human-labeled query. The only label record in DocSense; written only by a human
    keystroke through scripts/label_eval.py (§2 scripts) or by hand-editing the file."""
    query_id: str            # "q001"-style, unique within the file
    query: str               # non-blank after strip (validator)
    query_type: Literal["factoid", "definition", "method", "comparison",
                        "numeric_result", "table", "figure", "multi_hop"]
    difficulty: Literal["easy", "medium", "hard"]
    unanswerable: bool       # True => no span may have relevance >= 1 (validator)
    spans: list[LabeledSpan] # if not unanswerable, >= 1 span with relevance >= 1 (validator)
    tags: list[str]
    labeled_by: str          # non-blank; who made the judgment
    labeled_at: str          # ISO 8601 UTC
    pooled_from: list[str]   # retriever names whose candidates the labeler saw; [] = fully manual
    pool_k: int | None       # pool depth the labeler reviewed; makes pooling bias auditable (§5)

class CreditedHits(BaseModel):
    """One query's ranked hits resolved against its spans -- the only bridge between retrieval
    output and the pure arithmetic in metrics.py."""
    gains: list[int]                      # per rank: max relevance of spans NEWLY covered there, else 0
    first_covered_ranks: list[int | None] # per relevant span (input order): rank that first covered it
    ideal_gains: list[int]                # relevances of the relevant spans, descending

class QueryScore(BaseModel):
    query_id: str; query: str; query_type: str; difficulty: str
    num_relevant_spans: int; unanswerable: bool; scored: bool
    metrics: dict[str, float]             # "recall@5" / "mrr" / "ndcg@5"; {} when not scored
    first_relevant_rank: int | None       # 0-based, None if nothing relevant in the retrieved list
    retrieved: list[RetrievedRef]         # [] unless eval.per_query_detail

class RetrievedRef(BaseModel):
    """A retrieved hit reduced for the report: ids, pages, score, credit. Carries NO chunk text --
    corpus text must not land in a committable report (licensing, Phase 1 decision 3) and a report
    holding (query, text) pairs is label-shaped (§5)."""
    chunk_id: str; arxiv_id: str; page_start: int; page_end: int
    rank: int; score: float; credited_relevance: int

class EvalReport(BaseModel):
    """One eval run. The "log model and data versions with every eval run" record."""
    run_id: str              # f"{started_at:%Y%m%dT%H%M%SZ}_{variant_name}"
    started_at: str; finished_at: str
    gold_path: str; gold_sha256: str      # labels are a versioned input; a number is meaningless without it
    num_queries: int; num_scored: int; num_unanswerable: int
    retriever: str; retriever_config: str; variant_name: str
    embedder_model: str; index_fingerprint: str
    k_values: list[int]; primary_k: int; binary_relevance_threshold: int; retrieval_top_k: int
    seed: int; library_versions: dict[str, str]
    metrics: dict[str, float | None]      # None when num_scored == 0 (never 0.0 -- §5)
    metrics_by_slice: dict[str, dict[str, float | None]]  # "query_type=table" -> metrics
    mean_latency_ms: float | None
    per_query: list[QueryScore]
```

### `src/docsense/eval/metrics.py` — pure arithmetic, no chunk/span knowledge

Deliberately takes only sequences of numbers so every value is hand-checkable (§6.1).

```python
def dcg(gains: Sequence[float]) -> float:
    """sum(g_i / log2(i + 2)) over 0-based ranks -- the standard discount."""
def recall_at_k(first_covered_ranks: Sequence[int | None], k: int) -> float:
    """Fraction of relevant spans first covered at a rank < k ("span recall", Tradeoffs #2).
    Raises ValueError on an empty sequence (recall is undefined with no relevant spans) or k < 1."""
def reciprocal_rank(gains: Sequence[int], k: int, threshold: int = 1) -> float:
    """1/(r+1) for the first 0-based rank r < k with gains[r] >= threshold; 0.0 if there is none
    (Tradeoffs #3). Raises ValueError on k < 1."""
def ndcg_at_k(gains: Sequence[int], ideal_gains: Sequence[int], k: int) -> float:
    """dcg(gains[:k]) / dcg(sorted(ideal_gains, reverse=True)[:k]). Raises ValueError if
    ideal_gains is empty or sums to 0, or k < 1. Neither argument is re-sorted: `gains` is the
    retriever's order and ties are the retriever's to break (Tradeoffs #5)."""
def mean_or_none(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or None for an empty sequence -- an unmeasured metric is never 0.0."""
```

### `src/docsense/eval/relevance.py` — span matching and credit assignment

```python
def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Inclusive page-interval intersection test."""
def relevant_spans(example: GoldExample, threshold: int) -> list[LabeledSpan]:
    """Spans with relevance >= threshold, in file order. Excludes relevance-0 judgments."""
def credit_hits(hits: Sequence[RetrievedChunk], spans: Sequence[LabeledSpan],
                threshold: int) -> CreditedHits:
    """Resolve ranked hits against labeled spans (Tradeoffs #4, first-credit-only).

    A hit covers a span when arxiv_ids match and page intervals overlap. Walking ranks in order,
    each rank's gain is the max relevance of the spans it covers that no earlier rank covered
    (0 if it covers none); every span it newly covers records this rank. A span therefore pays out
    exactly once, so nDCG cannot be inflated by near-duplicate chunks of the same passage and
    recall counts spans rather than chunks. Raises ValueError if `hits` ranks are not 0..n-1.
    """
```

### `src/docsense/eval/io.py`

```python
def read_gold_set(path: Path, store: RecordStore) -> list[GoldExample]:
    """Read and validate every GoldExample from `path`; [] if the file is absent or empty (the
    end-of-phase state). Raises ValueError on a duplicate query_id or a duplicate
    whitespace-normalized, case-folded query, and pydantic.ValidationError on a malformed line."""
def write_eval_report(report: EvalReport, runs_dir: Path) -> Path:
    """Write runs_dir/<run_id>.json (model_dump_json, indent=2). Raises ValueError if the target
    resolves to the configured gold_path or to any *gold* filename (§5 leakage guard)."""
def export_eval_texts(gold: list[GoldExample], chunk_meta: list[IndexedChunk],
                      out_path: Path, store: RecordStore) -> int:
    """Write the eval-side input for `eval-leakage-check`: one record per gold query
    ({"kind": "query", "query_id", "text": query}) and per chunk overlapping any labeled span
    ({"kind": "passage", "query_id", "chunk_id", "text": chunk.text}), so
    `check_eval_leakage.py --field text` catches both leaked passages and leaked queries. Returns
    the record count. Derived and gitignored; never a label source."""
```

There is deliberately **no `write_gold_set`** anywhere in `src/docsense/` (§6 criterion 11).

### `src/docsense/eval/harness.py`

```python
def score_example(example: GoldExample, result: RetrievalResult, k_values: Sequence[int],
                  threshold: int, include_detail: bool) -> QueryScore:
    """Score one query. Returns scored=False with metrics={} when `example.unanswerable` (no
    relevant span exists, so all three metrics are undefined -- Tradeoffs #3); such queries are
    counted in the report and excluded from every mean."""
def evaluate(retriever: Retriever, gold: list[GoldExample], config: dict,
             retriever_config_name: str) -> EvalReport:
    """Run `retriever.retrieve_batch` over every gold query at eval.retrieval_top_k, score each,
    and aggregate. metrics are means over scored queries only (None when none are scored, which is
    exactly the empty-gold-set case). metrics_by_slice repeats the aggregation per value of each
    field in eval.slice_by. Raises ValueError if eval.retrieval_top_k < max(eval.k_values)."""
def run_eval(eval_config: dict, retriever_config: dict, retriever_config_name: str,
             index_dir: Path | None = None, gold_path: Path | None = None) -> EvalReport:
    """Load the gold set, load a DenseRetriever from `index_dir` (default the retriever config's
    index_dir), call `evaluate`, write the report, return it. Does not exit the process."""
```

### Scripts

- **`scripts/eval.py`** — `--config eval [--retriever-config NAME] [--index-dir DIR] [--gold PATH]
  [--out PATH] [--json] [--allow-empty]`. Prints one row per metric plus the slice table, writes
  the report under `eval.runs_dir`. Exit 0 on success; **3** when the gold set has zero scorable
  queries unless `--allow-empty` (so a stale/empty label file cannot be mistaken for a passing
  eval); 1 on a missing index or gold file. Warns when `num_scored < eval.min_queries_warn`.
- **`scripts/label_eval.py`** — the labeling scaffold. `--config eval --labeled-by NAME
  [--retriever-config NAME] [--pool-k N] [--query-id-prefix q] [--suggest-papers N]`.
  `--suggest-papers N` prints N `data/manifest.jsonl` rows (arxiv_id, title, query_category) sampled
  with `seed` for the human to read and write queries about — it emits no labels. Otherwise: read
  `labeling.queries_path` (human-authored, one query per line, `#` comments), skip queries already
  present in the gold set, and for each remaining query retrieve `labeling.pool_k` candidates and
  present each as `format_citation` + a `labeling.snippet_chars` snippet, prompting for one of
  `0|1|2` (grade), `s` (skip candidate), `a` (add a span by hand, e.g. `2301.00001:4-6 2`, for a
  passage retrieval missed), `u` (mark the query unanswerable), `n` (next query), `q` (save and
  quit). Blank or unrecognized input **re-prompts** — there is no default grade and no bulk path.
  Per query it then prompts for `query_type` and `difficulty` and appends one `GoldExample` with
  `labeled_by`, `labeled_at`, `pooled_from=[retriever.name]`, `pool_k`. Append-only: it never
  rewrites or deletes an existing line, so re-labeling is a deliberate human file edit. Prompting
  goes through an injectable `prompt_fn: Callable[[str], str] = input` so tests drive it with a
  scripted key sequence — that injection point is the *only* non-`input` path, so no code can label
  without a per-candidate human decision.
- **`scripts/export_eval_passages.py`** — `--config eval [--index-dir DIR] [--out PATH]`; wraps
  `export_eval_texts`, defaulting `--out` to `data/eval/derived/eval_texts.jsonl`. Prints the exact
  `check_eval_leakage.py` command to run next.

### Changes to existing files

- `src/docsense/retrieve/base.py`: add `retrieve_batch(self, queries: list[str], top_k: int | None
  = None) -> list[RetrievalResult]` to the `Retriever` protocol. `DenseRetriever` already
  implements it and the Phase 2 doc already named it "what Phase 3 uses"; putting it on the protocol
  keeps the harness off `hasattr`.
- `.gitignore`: add `data/eval/runs/` and `data/eval/derived/`. `data/eval/gold_set.jsonl` stays
  tracked (Phase 1 decision 3 anticipated this).
- Committed data files: `data/eval/gold_set.jsonl` (**zero bytes**) and `data/eval/README.md` (the
  §7 labeling protocol, the file format, and the `label_eval.py` key reference). No queries file and
  no example gold line ships — a plausible-looking example is one copy-paste away from being a fake
  label.

## 3. Data flow

```
data/eval/queries.txt        (COMMITTED, human-written, one query per line)
data/index/<variant>/        (Phase 2: vectors + chunk_meta + index_manifest)
      |  scripts/label_eval.py: pool_k candidates per query -> HUMAN keystrokes
      v
data/eval/gold_set.jsonl     (COMMITTED, append-only, one GoldExample per line)
      |  read_gold_set() -> list[GoldExample]
      |  DenseRetriever.retrieve_batch(queries, top_k=eval.retrieval_top_k)
      |  credit_hits() -> CreditedHits -> metrics.py -> QueryScore
      v
data/eval/runs/<run_id>.json (GITIGNORED EvalReport; ids/pages/scores only, no chunk text)
data/eval/derived/eval_texts.jsonl (GITIGNORED; feeds scripts/check_eval_leakage.py in Phase 4)
```

**Formats.** `gold_set.jsonl` is newline-delimited JSON read through `RecordStore`, one
`GoldExample` per line, append-only so a label's git history is a clean audit trail; it is the one
file in `data/` besides `manifest.jsonl` that is committed, because labels are the project's
irreplaceable human artifact. `queries.txt` is plain text (not JSONL) so the human can brainstorm in
an editor without touching JSON. `<run_id>.json` is one pretty-printed `EvalReport`. Reports are
gitignored run artifacts, like Phase 2's index: they are reproducible from `gold_sha256` +
`index_fingerprint` + the config, and Phase 7 promotes the specific reports it reports on into
`docs/results/` by an explicit human copy rather than accumulating hundreds of committed runs.

## 4. Config keys (`config/eval.yaml`)

One eval config across all variants: `retriever_config` names the variant config file to load, so a
Phase 7 ablation variant stays a single retrieval config copy instead of also duplicating the
embedder block here.

| Key | Default | Meaning |
|---|---|---|
| `retriever_config` | `"baseline"` | `config/<name>.yaml` stem of the retrieval variant to evaluate |
| `seed` | `0` | Logged in the report; seeds `--suggest-papers` sampling |
| `eval.gold_path` | `"data/eval/gold_set.jsonl"` | Committed human labels |
| `eval.runs_dir` | `"data/eval/runs"` | Report output (gitignored) |
| `eval.record_format` | `"jsonl"` | `RecordStore` backend for the gold set |
| `eval.k_values` | `[1, 3, 5, 10]` | k for Recall@k and nDCG@k |
| `eval.primary_k` | `5` | The headline k (matches `retrieval.top_k`) |
| `eval.retrieval_top_k` | `10` | Depth retrieved for scoring; must be >= `max(k_values)` |
| `eval.binary_relevance_threshold` | `1` | Relevance counted as relevant for Recall/MRR |
| `eval.per_query_detail` | `true` | Include `QueryScore.retrieved` in the report |
| `eval.slice_by` | `["query_type", "difficulty"]` | Fields to break metrics down by |
| `eval.min_queries_warn` | `30` | Warn below this many scored queries |
| `labeling.queries_path` | `"data/eval/queries.txt"` | Human-authored query list |
| `labeling.pool_k` | `20` | Candidates shown per query (> `retrieval_top_k` on purpose) |
| `labeling.snippet_chars` | `400` | Snippet length in the labeling prompt |

## 5. Tradeoffs

1. **Label unit: chunk_ids vs page spans vs both.** Page spans (`arxiv_id` + page range).
   chunk_ids are a function of `chunk_size_tokens`, so a chunk_id-keyed gold set would silently
   decay the moment Phase 7 ablates chunking — the one thing that must survive every variant.
   Spans are chunking-invariant and already the citation unit. Cost: page granularity, so a
   retrieved chunk that merely touches a relevant page scores as relevant; accepted because page
   numbers are what DocSense actually promises a user. `source_chunk_id` is retained as provenance
   only, never scored.
2. **Recall@k definition: chunk recall vs span recall.** Span recall — fraction of *labeled spans*
   covered by the top-k. Chunk recall needs a complete chunk-level ground truth (every chunk
   overlapping a span), which is chunking-dependent and inflates the denominator with
   near-duplicate chunks the human never saw. Span recall's denominator is exactly what the human
   labeled.
3. **Queries with no relevant document retrieved, and unanswerable queries.** Two different cases,
   handled differently. Nothing relevant in the top-k: `reciprocal_rank = 0.0`, recall `0.0` — the
   query is scored, because "found nothing" is a real failure and dropping it would flatter the
   retriever. A query the human labeled `unanswerable`: **excluded** from all three means (they are
   undefined without a relevant span) and reported as `num_unanswerable`; it becomes an abstention
   metric when generation lands. The `unanswerable` flag is required precisely so a mislabeled or
   half-finished example cannot masquerade as either case.
4. **Multiple relevant spans and duplicate coverage: standard nDCG vs first-credit-only.**
   First-credit-only: a span pays out once, at the highest rank covering it. Textbook nDCG gives
   every relevant item its gain, so three overlapping chunks of the same passage triple-count and a
   degenerate retriever that returns near-duplicates of one hit scores near 1.0 — a real failure
   mode for a chunked corpus with 50-token overlaps. `ideal_gains` is correspondingly the span
   relevances sorted descending, so nDCG@k = 1.0 means "the top ranks cover distinct relevant
   spans, best first". This is a documented deviation from textbook nDCG; it is stated in the
   `credit_hits` docstring and in `data/eval/README.md` so no reader mistakes it for the standard.
5. **Ties in ranking.** Metrics never re-sort: they consume the retriever's rank order, and ties are
   broken upstream by FAISS/argpartition row order. Re-sorting inside the metric would hide a
   retriever's own nondeterminism; instead §6 asserts the metrics are order-preserving, and Phase 7
   keeps `seed` and `index_fingerprint` in every report so a tie-order change is traceable.
6. **Grades: binary vs graded 0/1/2.** Graded, with `binary_relevance_threshold` collapsing it for
   Recall/MRR. The marginal labeling cost is one extra keystroke decision per candidate, and it is
   the difference between an nDCG that measures ordering quality and one that duplicates Recall.
   `0` is a *recorded* judgment (the human saw it and said no), which is what makes the pool
   auditable.
7. **Labeling workflow: pool-and-judge vs read-paper-and-cite vs generated queries.** Generated
   queries are forbidden (CLAUDE.md) and would also be answerable-by-construction. Pure
   read-and-cite is unbiased but slow. Pick: human writes queries from papers they chose
   (`--suggest-papers` for a seeded sample), then judges a `pool_k=20` pool, **plus** an `a`
   command to add a span the pool missed. Honest cost — **pooling bias**: labels drawn from the
   baseline's pool can only credit what the baseline found, which biases Recall upward and
   understates Phase 4's gains. Mitigations: pool depth 20 > eval depth 10; the manual-add command;
   `pooled_from` and `pool_k` recorded per example so the bias is auditable; and Phase 4 re-pools
   with the hybrid retriever and re-labels the union, which is the standard TREC remedy and must be
   budgeted there rather than discovered.
8. **Report location: committed vs gitignored.** Gitignored under `data/eval/runs/`, matching
   Phase 2's treatment of the index: reproducible, and auto-committing every run would bury the one
   artifact that matters. Phase 7 promotes chosen reports to `docs/results/` by hand. Reports carry
   no chunk text, so promoting one raises no licensing question.

**Eval leakage.** This phase creates the eval set, so it is the phase with the most ways to corrupt
it. (a) **Fabrication.** No function in `src/docsense/` writes the gold set; the only writer is
`scripts/label_eval.py`, which appends one example per completed human session and has no default
grade, no "accept all", and no inference from rank — asserted by §6 criterion 11. (b) **Label-shaped
artifacts.** An `EvalReport` is a query paired with chunk ids and relevance grades, i.e. exactly a
gold line; `write_eval_report` therefore refuses to write to `eval.gold_path` or any `*gold*`
filename, complementing Phase 2's `query.py --out` guard, and `RetrievedRef` carries no text so a
report can never be reinterpreted as a labeled passage. (c) **Relevance-0 spans are not training
negatives.** They are attached to eval queries; mining them as Phase 4 hard negatives would train on
the eval set. Documented in the `LabeledSpan` docstring and in `data/eval/README.md`. (d) **Phase 4
training data.** `export_eval_texts` is the sanctioned eval-side input to
`scripts/check_eval_leakage.py`, covering both labeled passage text and gold query text; Phase 4
must exclude any paper appearing in the gold set from synthetic-query generation and then run the
`eval-leakage-check` skill. (e) **Tuning on eval.** Labeling must finish before any variant is
compared, and labels must not be revised after seeing a variant's score; `gold_sha256` in every
report makes a post-hoc label edit visible in the git history and in the number it produced.

## 6. Acceptance criteria

### 6.1 Hand-computed metric fixture (the reviewer's first check)

With `gains = [0, 2, 0, 1]`, `first_covered_ranks = [1, 3, None]`, `ideal_gains = [2, 1, 1]`,
`k = 4`, `threshold = 1`, tests must assert (`pytest.approx`, rel=1e-6):

| Quantity | Expected | Working |
|---|---|---|
| `recall_at_k(..., 1)` | `0.0` | no span covered at rank 0 |
| `recall_at_k(..., 4)` | `2/3` ≈ `0.666667` | spans covered at ranks 1 and 3; third never |
| `reciprocal_rank(gains, 4)` | `0.5` | first gain >= 1 at 0-based rank 1 |
| `reciprocal_rank(gains, 1)` | `0.0` | nothing relevant within k=1 |
| `dcg([2, 1, 1])` | `3.130930` | `2/1 + 1/log2 3 + 1/log2 4` |
| `dcg(gains)` | `1.692536` | `2/log2 3 + 1/log2 5` |
| `ndcg_at_k(gains, ideal_gains, 4)` | `0.540586` | `1.692536 / 3.130930` |

1. The table above passes exactly as written, in `tests/test_eval_metrics.py`.
2. Metric edge cases: `ndcg_at_k(ideal_gains, ideal_gains, 3) == 1.0`; `ndcg_at_k([1, 0], [1], 2)
   == 1.0`; a reversed ranking scores strictly lower than the ideal one; `k` exceeding the list
   length does not raise or index out of range; `recall_at_k([], k)`, `ndcg_at_k(g, [], k)`,
   `ndcg_at_k(g, [0, 0], k)` and any `k < 1` raise `ValueError`; `mean_or_none([]) is None`.
3. Metrics are order-preserving: `ndcg_at_k` and `reciprocal_rank` on a permuted `gains` return a
   different value (no internal sort), and neither mutates its arguments.
4. `credit_hits` (`tests/test_eval_relevance.py`): on hand-built `RetrievedChunk`s and
   `LabeledSpan`s it credits a hit whose page range overlaps a span, does not credit a different
   `arxiv_id` with the same pages, gives a second hit covering an already-credited span gain `0`,
   gives a hit covering two uncovered spans gain `max(relevance)` while recording a
   `first_covered_rank` for *both*, ignores relevance-0 spans, returns `ideal_gains` descending,
   and raises `ValueError` on non-contiguous hit ranks. Boundary case: spans `[4,5]` and `[5,6]`
   both count as covered by a hit on page 5.
5. Schema round-trip (`tests/test_eval_schema.py`): every model round-trips `model_dump_json()` ->
   `model_validate_json()` to an equal object, including a `GoldExample` with multiple spans and an
   `unanswerable` one with `spans=[]`.
6. Schema validators reject: `page_start > page_end`; `relevance` of `-1` or `3`; a blank or
   whitespace-only `query`; a blank `labeled_by`; `unanswerable=True` with a relevance-1 span;
   `unanswerable=False` with `spans=[]` or with only relevance-0 spans.
7. `read_gold_set` (`tests/test_eval_io.py`): round-trips hand-written lines in file order; returns
   `[]` for a missing file and for a zero-byte file; raises `ValueError` on a duplicate `query_id`
   and on the same query text differing only in case/whitespace; raises `ValidationError` on a line
   missing `labeled_by`.
8. **The committed gold set is empty at phase end:** `data/eval/gold_set.jsonl` exists, is zero
   bytes, and `read_gold_set` on it returns `[]` — asserted by a test reading the real repo path.
9. `evaluate` on an empty gold set (`tests/test_eval_harness.py`) returns an `EvalReport` with
   `num_queries == num_scored == 0`, every value in `metrics` `None` (not `0.0`), `per_query == []`,
   and populated provenance (`gold_sha256`, `index_fingerprint`, `variant_name`, `seed`,
   `library_versions`) — no exception.
10. `evaluate` end-to-end on a **synthetic in-test** gold set (3-4 `GoldExample`s written in test
    code against the `conftest` fixture corpus, hash embedder + numpy store): metric keys are
    exactly `recall@k`/`ndcg@k` for each `k_values` entry plus `mrr`; `num_scored` excludes the
    unanswerable example and `num_unanswerable == 1`; a query whose only labeled span is the top hit
    scores `recall@1 == mrr == ndcg@1 == 1.0`; a query labeled on a span no chunk covers scores
    `0.0` on all three and `first_relevant_rank is None`; `metrics_by_slice` has a key per observed
    `query_type` and `difficulty`; `retrieval_top_k < max(k_values)` raises `ValueError`.
11. **Fabrication guards** (`tests/test_eval_no_label_writes.py`): no name matching
    `write_gold|append_gold|generate_gold|synthes` exists in `src/docsense/eval/`; no module under
    `src/docsense/` references `gold_set` as a write target (source scan for `open(..., "w"/"a")`
    and `write_text` near the gold path); `write_eval_report` raises `ValueError` for a `runs_dir`
    whose target path is the gold path or any `*gold*` filename; and `RetrievedRef` has no `text`
    field.
12. `scripts/label_eval.py` (`tests/test_scripts_label_eval.py`), driven by a scripted `prompt_fn`
    against the fixture index: a session grading two candidates `2` and `1`, answering `n`, then
    `q`, appends **exactly one** line whose `GoldExample` has those two spans with
    `source_chunk_id` set, `labeled_by` from the flag, `pooled_from == ["dense"]`, and `pool_k`
    recorded. Separately: blank and garbage input re-prompt instead of defaulting to a grade
    (counted prompt calls); `u` writes `unanswerable=True` with no relevance>=1 span; `a` with
    `2301.00001:4-6 2` adds a span with `source_chunk_id is None`; an existing line in the file is
    byte-identical after the run and a query already in the gold set is skipped; exhausting the
    prompt sequence without a completed query appends nothing; a missing `queries_path` and a
    missing `--labeled-by` each exit non-zero with a message pointing at `data/eval/README.md`;
    `--suggest-papers 3` prints 3 manifest rows, is deterministic under `seed`, and writes nothing.
13. `scripts/eval.py` (`tests/test_scripts_eval.py`) with `--retriever-config offline_smoke` against
    a `tmp_path` fixture corpus: exits 3 on an empty gold set, exits 0 with `--allow-empty`, exits 0
    and writes exactly one `runs_dir/<run_id>.json` validating as `EvalReport` on a small synthetic
    gold set, exits 1 on a missing index dir and on a missing `--gold` path, prints the
    `min_queries_warn` warning, and `--json` emits parseable JSON.
14. `scripts/export_eval_passages.py` writes one `{"kind": "query"}` record per gold query and one
    `{"kind": "passage"}` record per chunk overlapping a labeled span, each with a `text` field, and
    `scripts/check_eval_leakage.py --eval <that file> --field text` runs against it and detects a
    deliberately copied passage (links the two tools end-to-end).
15. `load_config("eval")` succeeds and its key set matches §4 exactly (config-completeness test, so
    an ablation copy cannot silently drop a key); `eval.retrieval_top_k >= max(eval.k_values)` and
    `eval.primary_k in eval.k_values` hold for the shipped file.
16. `.gitignore` ignores `data/eval/runs/` and `data/eval/derived/` but **not**
    `data/eval/gold_set.jsonl` or `data/eval/README.md` (extend `tests/test_gitignore.py`);
    `Retriever` declares `retrieve_batch` and `DenseRetriever` still satisfies the protocol.
17. No test downloads weights or makes a network request (the `conftest` guard still applies), no
    test writes to the real `data/eval/gold_set.jsonl` (gold paths in tests are `tmp_path`), and
    `data/eval/README.md` documents: the §7 human protocol, the file format field by field, the
    `label_eval.py` key reference, the first-credit nDCG deviation (Tradeoffs #4), and the
    relevance-0-is-not-a-negative rule (§5c).
18. `check` skill (pytest -q + ruff check .) passes with these tests added.

## 7. The human labeling step (after implementation, before any reported number)

Implementation ends with an empty gold set. The human then, in one or more sittings:

1. `python scripts/build_index.py --config baseline` (Phase 2) if the index is not already built.
2. `python scripts/label_eval.py --config eval --suggest-papers 20` -> skim those papers.
3. Write questions into `data/eval/queries.txt`, one per line, phrased as a real user would ask
   (not copied from a paper's sentences, which would make retrieval trivially lexical). Aim for a
   spread across `query_type` — including `table` and `figure` queries, which Phase 5 needs — and
   across the four `fetch.categories`.
4. `python scripts/label_eval.py --config eval --labeled-by "<name>"` and judge each pool, using
   `a` whenever a known-relevant passage is missing from the pool.
5. `python scripts/eval.py --config eval` for the first real baseline numbers, and commit
   `data/eval/gold_set.jsonl` and `data/eval/queries.txt`.

**Recommended effort** (for the human's §8 decision): >= 50 queries for the headline numbers, 80-120
for slice breakdowns to mean anything, with 2-4 relevant spans per query and ~10% deliberately
`unanswerable`. Budget roughly 3-6 minutes per query (writing + judging 20 candidates). The
statistical consequence must be stated in Phase 7: at n=50 the standard error on Recall@5 is roughly
0.07, so only differences above ~0.10 are defensible, and Phase 7 should report a paired bootstrap
or sign test over per-query scores rather than two bare means.

## 8. Decisions and open questions

**Decided autonomously this round** (routine design calls delegated by the human; reviewer should
treat as fixed): page spans as the label unit over chunk_ids (Tradeoffs #1); span-based Recall@k
(#2); unanswerable queries excluded from means while zero-hit queries are scored (#3);
first-credit-only nDCG with span-derived `ideal_gains` (#4); no re-sorting in metrics (#5); graded
0/1/2 with a config binary threshold (#6); pool-and-judge with a manual-add escape plus recorded
`pooled_from`/`pool_k` (#7); gitignored run reports with human promotion in Phase 7 (#8);
`data/eval/gold_set.jsonl` as the committed path; one `config/eval.yaml` referencing
`retriever_config` rather than an eval config per variant; `retrieve_batch` promoted onto the
`Retriever` protocol; required `query_type`/`difficulty` enums so Phase 5's multimodal slice needs
no re-labeling.

**Open — needs the human, and only the human.**

1. **How many gold queries to label, and when.** This is a scheduling and effort question, not a
   design question: the harness works at any n, and §7 recommends >= 50 with the n=50 statistical
   caveat, but the human owns the time budget, the target count, and whether Phase 4 starts before
   labeling finishes (it must not start *comparing variants* before then — §5e). A related judgment
   that comes with it: the house style for query phrasing (terse keyword-ish vs full natural
   questions), which shifts what the retrieval numbers mean and is a taste call.
2. **Nothing else is open.** No hardware, API key, or corpus-scope decision is required by this
   phase: evaluation reuses the Phase 2 index and embedder, calls no external API, and adds no
   dependency.
