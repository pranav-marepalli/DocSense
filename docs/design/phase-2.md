# Phase 2: Baseline RAG — retrieval half

## 1. Goal

Turn `data/parsed/chunks.jsonl` into a queryable dense index and answer queries with ranked,
page-cited passages. Two parts: (a) **embed + index** the corpus with one sentence-transformers
model into a flat exact-search vector store, writing a self-contained index directory plus a
provenance manifest; (b) a **single dense retriever** mapping a query to ranked `RetrievedChunk`s
and resolving each to a `Citation` carrying Phase 1's `arxiv_id`/`page_start`/`page_end` verbatim.
This is deliberately the *non-hybrid* baseline — no BM25, no reranker, no fine-tuning, no
multimodal (Phases 4 and 5) — and exists so those phases have something to beat. **Generation is
deferred, not dropped** (decision 1): no LLM is called in this phase, `src/docsense/generate/`
stays an empty package, and §2's "Deferred" subsection records the answer-synthesis contract so a
later phase can pick it up once a provider is chosen and funded. The consequence to carry into
Phase 3: the eval harness scopes to *retrieval* metrics (Recall@k, nDCG@k, MRR) until then.
**Measured by:** every acceptance criterion in §6 passing with `check` green, and
`scripts/query.py` running end-to-end over a fixture corpus. No retrieval *quality* numbers here —
no gold set exists until Phase 3, and nothing in this phase may be tuned against one (§5, leakage).

## 2. Module plan

New files under `src/docsense/index/` and `src/docsense/retrieve/` (both currently empty
`__init__.py`), two CLI scripts, two configs, `pyproject.toml` and `.gitignore` changes.

### Schemas — pydantic models, no I/O, one module per package (the Phase 1 pattern)

`src/docsense/index/schema.py`:

- **`IndexedChunk`** — one sidecar row: a Phase 1 `Chunk` plus its vector row, written in `row_id`
  order so lookup is list indexing rather than a scan. `row_id: int`, then `chunk_id: str`,
  `arxiv_id: str`, `chunk_index: int`, `page_start: int`, `page_end: int`, `text: str`,
  `token_count: int`, `source: Literal["text"]` copied verbatim from the `Chunk`.
- **`EmbedderInfo`** — exactly which embedder produced an index; compared against config on load to
  reject mismatches: `type: Literal["sentence_transformer", "hash"]`, `model_name: str`, `dim: int`,
  `normalize: bool`, `max_seq_length: int | None`, `query_prefix: str`, `passage_prefix: str`.
- **`IndexManifest`** — one build, at `<index_dir>/index_manifest.json`; the "log model and data
  versions" record for retrieval: `variant_name: str`, `built_at: str` (ISO 8601 UTC), `seed: int`,
  `embedder: EmbedderInfo`, `index_type: str`, `num_vectors: int`, `num_chunks: int`,
  `num_papers: int`, `chunks_path: str` (relative to repo root), `chunks_sha256: str` (input corpus
  identity), `fingerprint: str` (sha256 over embedder + index_type + chunks_sha256 + num_chunks),
  `library_versions: dict[str, str]` (sentence_transformers, faiss, numpy, torch).

`src/docsense/retrieve/schema.py`:

- **`RetrievedChunk`** — one hit, carrying the citation fields verbatim (nothing downstream may
  re-derive page numbers from text): `chunk_id: str`, `arxiv_id: str`, `page_start: int`,
  `page_end: int`, `text: str`, `score: float`, `rank: int` (0-based, ascending with descending
  score).
- **`Citation`** — a `RetrievedChunk` reduced to its provenance, which is what Phase 3 scores and
  Phase 6 renders: `chunk_id: str`, `arxiv_id: str`, `page_start: int`, `page_end: int`,
  `rank: int`, `score: float`. Every field is copied from the chunk; nothing is ever parsed out of
  generated text.
- **`RetrievalResult`** — one query's hits plus the provenance to reproduce them: `query: str`,
  `retriever: str` (e.g. `"dense"`), `top_k: int`, `variant_name: str`, `embedder_model: str`,
  `index_fingerprint: str`, `latency_ms: float`, `results: list[RetrievedChunk]`.

### `src/docsense/index/embed.py`

```python
class Embedder(Protocol):
    """Text -> float32 [n, dim]. The only embedding interface callers may use."""
    info: EmbedderInfo
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...

class SentenceTransformerEmbedder:
    """sentence-transformers backend. Applies `query_prefix` when is_query else `passage_prefix`
    (bge/e5-style asymmetric models need this; prefixes are config, not code). float32, L2-
    normalized when `normalize`, so inner product == cosine. Loads the model lazily on first
    encode, so constructing one downloads nothing."""
    def __init__(self, model_name: str, dim: int | None, normalize: bool, batch_size: int,
                 device: str | None, max_seq_length: int | None, query_prefix: str,
                 passage_prefix: str, seed: int) -> None: ...

class HashEmbedder:
    """Deterministic stub: hashed bag-of-words projected to `dim`, L2-normalized. DEV AND TEST
    ONLY -- no semantics. Warns on construction and records type="hash" in EmbedderInfo, so any
    index built with it is visibly non-real."""
    def __init__(self, dim: int, normalize: bool, seed: int) -> None: ...

def get_embedder(config: dict) -> Embedder:
    """Factory over config['embedder']['type']. ValueError on an unknown type."""
```

### `src/docsense/index/store.py` — vector persistence, mirrors `ingest.io.RecordStore`

```python
class VectorStore(Protocol):
    """Exact top-k inner-product search over a fixed-dim matrix. One file on disk."""
    index_type: str; dim: int; filename: str
    def add(self, vectors: np.ndarray) -> None: ...
    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]: ...
    def save(self, path: Path) -> None: ...
    def __len__(self) -> int: ...

class FaissFlatIPStore:
    """faiss.IndexFlatIP. index_type='faiss_flat_ip', filename='vectors.faiss'.
    __init__ calls faiss.omp_set_num_threads(num_threads) (config index.num_threads, default 1)
    to avoid a macOS duplicate-libomp crash between torch's and faiss's bundled OpenMP runtimes
    in faiss's multi-query search()."""
    def __init__(self, dim: int, num_threads: int = 1) -> None: ...
class NumpyFlatIPStore:
    """Brute-force `vectors @ q.T` + argpartition. index_type='numpy_flat_ip',
    filename='vectors.npy'. Exact, so it ranks identically to FAISS up to float ties; used in
    tests and anywhere faiss will not install."""

def get_vector_store(index_type: str, dim: int, num_threads: int = 1) -> VectorStore:
    """Empty store by type. ValueError on an unknown index_type. num_threads is passed to
    FaissFlatIPStore only (see above); ignored for numpy_flat_ip."""
def load_vector_store(index_dir: Path, index_type: str, dim: int,
                       num_threads: int = 1) -> VectorStore:
    """Load from index_dir/<filename>. FileNotFoundError if absent; ValueError on a dim mismatch."""
```

`search` returns `(scores, row_ids)`, both `[n_queries, min(top_k, len(self))]`, rows sorted by
descending score. `add` raises `ValueError` on a non-float32 array or a dim mismatch.

### `src/docsense/index/build.py`

```python
def corpus_fingerprint(embedder_info: EmbedderInfo, index_type: str, chunks_sha256: str,
                       num_chunks: int) -> str:
    """sha256 over the canonical JSON of its arguments -- the rebuild-needed test."""
def build_index(chunks_path: Path, index_dir: Path, config: dict,
                force: bool = False) -> IndexManifest:
    """Embed every chunk in chunks_path and write a self-contained index directory:
    <index_dir>/{vectors.*, chunk_meta.jsonl, index_manifest.json}.

    Reads chunks via ingest.io.read_chunks (one source of truth for the Chunk schema), embeds them
    as passages in config['embedder']['batch_size'] batches, adds them in file order so row_id ==
    line number, and writes chunk_meta through the RecordStore named by config['record_format'].
    Idempotent: if index_manifest.json exists and its `fingerprint` matches the current inputs,
    returns it without re-embedding unless `force`.

    Raises:
        FileNotFoundError: if chunks_path does not exist.
        ValueError: if chunks_path holds no chunks (an empty index is never written).
    """
```

### `src/docsense/retrieve/base.py` and `dense.py`

```python
class Retriever(Protocol):
    """Query -> RetrievalResult. The seam Phase 4's hybrid/reranked retrievers and a later
    generation phase both plug into."""
    name: str
    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult: ...

class DenseRetriever:
    """Embed the query, exact top-k inner product, map row_ids back to IndexedChunks. No hybrid,
    no reranking (Phase 4)."""
    name: str = "dense"
    def __init__(self, store: VectorStore, chunk_meta: list[IndexedChunk], embedder: Embedder,
                 manifest: IndexManifest, top_k: int,
                 max_chunks_per_paper: int | None = None) -> None: ...
    @classmethod
    def load(cls, index_dir: Path, config: dict) -> "DenseRetriever":
        """Load manifest, store, and chunk_meta from index_dir; build the embedder from config.
        ValueError if the manifest's EmbedderInfo disagrees with config's (model_name, dim,
        normalize, prefixes) -- querying an index with a model other than the one that built it is
        silently wrong, so it is a hard error. FileNotFoundError if index_dir has no manifest."""
    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Ranked hits, top_k defaulting to the configured value. ValueError on an empty or
        whitespace-only query. With max_chunks_per_paper set, keeps only the highest-scoring n hits
        per arxiv_id before truncating to top_k, then renumbers ranks contiguously."""
    def retrieve_batch(self, queries: list[str],
                       top_k: int | None = None) -> list[RetrievalResult]:
        """One embedding call for all queries, same per-query semantics. What Phase 3 uses."""
```

### `src/docsense/retrieve/cite.py` — the page-level citation contract, retrieval side

```python
def citation_for(chunk: RetrievedChunk) -> Citation:
    """Project one hit onto its provenance. Pure field copying: an arXiv id or page number in a
    DocSense citation is always carried from the Chunk that Phase 1 extracted, never inferred."""
def citations_for(result: RetrievalResult) -> list[Citation]:
    """Citations for every hit, in rank order."""
def format_citation(citation: Citation) -> str:
    """Human/log rendering: "arXiv:2301.12345 p.4-5", collapsing to "p.4" when
    page_start == page_end."""
```

### Deferred: generation (decision 1)

No LLM is called in Phase 2 and `src/docsense/generate/` stays an empty package. Recorded so a
later phase can resume rather than redesign: generation should consume a `Retriever` via the §2
protocol, render hits as numbered `[n] arXiv:<id> pages <a>-<b>` context blocks under a token
budget, keep the prompt in `config/prompts/<name>.md` with its sha256 logged, and have the model
emit only `[n]` markers — so every page number in an answer is resolved from the retrieved chunk
through `citation_for`, and an out-of-range marker is detectably dropped rather than plausibly
wrong. An `Answer` record (answer text, resolved `Citation`s, abstention flag, LLM/model
provenance) and an OpenAI-compatible client with the key read from an env var belong to that phase,
alongside its own cost and reproducibility decisions.

### Scripts, configs, packaging

- `scripts/build_index.py`: `--config baseline [--chunks PATH] [--index-dir DIR] [--force]`; calls
  `build_index`, prints the manifest, exits non-zero if the corpus was empty.
- `scripts/query.py`: `--config baseline --query TEXT [--top-k N] [--index-dir DIR] [--json]
  [--out PATH]`; loads a `DenseRetriever`, prints each hit as `format_citation` plus a text snippet
  (or the full `RetrievalResult` JSON with `--json`). Exits non-zero on an empty query or a missing
  index. `--out` writes one `RetrievalResult` JSON and refuses any path under an `eval` directory
  (§5 leakage guard).
- `config/baseline.yaml` (§4); `config/offline_smoke.yaml` (same keys, `embedder.type: hash`,
  `index.type: numpy_flat_ip`) — dev/CI only, never a reported result.
- `pyproject.toml`: add `numpy` to `dependencies` (decision 5) — it is used directly and is
  currently only transitive via faiss/torch.
- `.gitignore`: add `data/index/`. `*.faiss` is already ignored, but `chunk_meta.jsonl` and
  `index_manifest.json` are not covered by any existing rule.

## 3. Data flow

```
data/parsed/chunks.jsonl  (Phase 1, gitignored)
      |  read_chunks() -> list[Chunk]
      |  Embedder.encode(passage_prefix + text) -> float32 [N, dim], L2-normalized
      v
data/index/<variant_name>/
      vectors.faiss | vectors.npy    one row per chunk; row_id == line number in chunks.jsonl
      chunk_meta.jsonl               one IndexedChunk per line, ascending row_id, via RecordStore
      index_manifest.json            IndexManifest, pretty-printed

query -> Embedder.encode(query_prefix + query, is_query=True) -> store.search(top_k)
      -> (scores, row_ids) -> chunk_meta[row_id] -> RetrievalResult(list[RetrievedChunk])
      -> citations_for -> list[Citation] -> printed, or one JSON object via --out
```

**Formats.** `chunk_meta.jsonl` is newline-delimited JSON written only via `RecordStore`, so Parquet
later is a config value (Phase 1 Tradeoffs #2). `index_manifest.json` is one pretty-printed
`IndexManifest`. `vectors.faiss` is `faiss.write_index`; `vectors.npy` is `np.save` of a float32
`[N, dim]` array. The whole directory is reproducible from `chunks.jsonl` plus the config, so none
of it is committed. `RetrievalResult`s are printed, or written only where the user asks (`--out`);
Phase 3 owns run logging. Scores are cosine similarities in `[-1, 1]`, comparable across variants.

## 4. Config keys (`config/baseline.yaml`)

The whole retrieval stack is one config file, so a Phase 4/7 ablation variant is a copy of this file
with fields changed — never a code change. `index_dir` is per-variant so variants cannot clobber
each other's indices. No `generation.*` block exists yet (decision 1).

| Key | Default | Meaning |
|---|---|---|
| `variant_name` | `"baseline"` | Logged into every manifest/result; names the index dir |
| `seed` | `0` | Seeds numpy/torch before embedding, and the hash embedder |
| `chunks_path` | `"data/parsed/chunks.jsonl"` | Phase 1 output to index |
| `index_dir` | `"data/index/baseline"` | Index output root |
| `record_format` | `"jsonl"` | `RecordStore` backend for `chunk_meta` |
| `embedder.type` | `"sentence_transformer"` | or `"hash"` (dev only) |
| `embedder.model_name` | `"BAAI/bge-small-en-v1.5"` | HF id; Phase 4's fine-tune base (decision 2) |
| `embedder.dim` | `384` | Asserted against the loaded model; required for `hash` |
| `embedder.normalize` | `true` | L2-normalize so inner product == cosine |
| `embedder.batch_size` | `32` | Encode batch size |
| `embedder.device` | `null` | Auto-detect (decision 3); `"cpu"`/`"mps"`/`"cuda"` to force |
| `embedder.max_seq_length` | `512` | Truncation length; `null` = model default |
| `embedder.query_prefix` | `"Represent this sentence for searching relevant passages: "` | bge-style query instruction |
| `embedder.passage_prefix` | `""` | Prefix for indexed passages |
| `embedder.trust_remote_code` | `false` | Never load remote code by default |
| `index.type` | `"faiss_flat_ip"` | or `"numpy_flat_ip"`; exact search only |
| `index.num_threads` | `1` | `faiss.omp_set_num_threads` argument (faiss stores only); avoids a macOS duplicate-libomp crash between torch's and faiss's bundled OpenMP runtimes when faiss's multi-query `search()` runs with its default thread count after torch has imported. Raise it if your environment is known not to hit that conflict, for more search parallelism |
| `retrieval.top_k` | `5` | Hits returned (decision 4) |
| `retrieval.max_chunks_per_paper` | `null` | `null` = no cap; diversity knob for Phase 7 |

## 5. Tradeoffs

1. **Embedding model: `bge-small-en-v1.5` vs `all-MiniLM-L6-v2` vs a base-size model.** bge-small:
   same 384 dims and CPU cost as MiniLM, materially better retrieval, small enough for Phase 4 to
   fine-tune on a laptop. Cost: it is asymmetric and needs a query instruction prefix, which is why
   prefixes are config fields rather than hardcoded. Base-size is ~3x slower per chunk for a modest
   gain — a Phase 7 ablation, not a baseline. MiniLM stays selectable in two config lines.
2. **Vector store: FAISS `IndexFlatIP` vs numpy brute force vs HNSW/IVF.** ~300 papers is roughly
   10-20k chunks, where exact search is milliseconds; approximate indexes would add recall error and
   tuning knobs that confound Phase 7's ablation without buying speed. FAISS flat is the default
   (already a dependency, importable on this interpreter); `NumpyFlatIPStore` is a second
   implementation behind `VectorStore` so tests need no faiss and a faiss-less environment still
   works. Both exact, so they rank identically.
3. **Cosine via normalized vectors + IP vs `IndexFlatL2`.** Normalize and use IP: scores land in a
   bounded, model-independent range, so they are printable, thresholdable, and comparable across
   Phase 7 variants. L2 distances are none of those.
4. **Embedding persistence: no cache vs per-chunk cache vs manifest fingerprinting.** Fingerprint:
   `build_index` hashes (embedder + index type + corpus sha + chunk count) into the manifest and
   skips re-embedding on a match. A full build is minutes on CPU, and a per-chunk cache keyed by
   text is the one place staleness could silently corrupt an ablation result. `--force` rebuilds.
5. **Index metadata: self-contained sidecar carrying `text` vs joining `chunks.jsonl` per query.**
   Self-contained: retrieval and Phase 6 serving load one directory, and the index cannot drift from
   a `chunks.jsonl` re-ingested underneath it. Cost: text duplicated (tens of MB, gitignored), with
   `chunks_sha256` keeping provenance auditable.
6. **Citations: projected from the chunk vs reconstructed downstream.** `Citation` is pure field
   copying from `RetrievedChunk` (itself copied from the Phase 1 `Chunk`), so page numbers are
   structurally incapable of being invented, and the same function serves Phase 3 scoring, Phase 6
   rendering, and a later generation phase. The alternative — letting a generator emit ids and pages
   as text — is the main hallucination vector and is why the deferred design keeps the model to
   `[n]` markers only.
7. **Scope: ship retrieval + generation together vs retrieval only (decision 1).** Retrieval only.
   Generation needs a funded provider decision, and coupling the two would have blocked Phase 3's
   retrieval metrics on that decision. Cost: Phase 3 cannot measure answer quality yet, and the
   `Retriever` protocol has only one implementation until Phase 4, so the seam is unproven until
   then. The deferred-generation sketch in §2 keeps the work recoverable.
8. **Test double (`HashEmbedder`) in `src/` vs importorskip-and-download.** A real double makes the
   whole §6 checklist offline and keeps `check` fast, and `offline_smoke.yaml` demonstrates the
   config-per-variant pattern immediately. Risk mitigated by provenance: it warns on construction,
   and `embedder.type == "hash"` makes an accidental run visible in its manifest.

**Eval leakage.** No training data or labels are produced here, so there is no leakage *mechanism* —
but two live risks. (a) **Tuning on eval:** the gold set does not exist yet, and nothing here
(`top_k`, chunk caps, the embedder choice) may later be tuned by inspecting gold relevance
judgments; smoke queries stay in `docs/` or test fixtures and must never be copied into Phase 3's
gold file. (b) **Outputs vs labels:** a `RetrievalResult` is literally the shape of a relevance
label — a query paired with chunk ids — so writing one into `data/eval/` or anywhere a labeling
tool reads would fabricate the gold set, which CLAUDE.md forbids. Hence `scripts/query.py --out`
must refuse any path under an `eval` directory, and Phase 3's gold judgments must be produced by
the human before any retrieval output is inspected per query. Forward-looking:
`chunk_meta.jsonl` holds verbatim corpus text, so Phase 4's synthetic training pairs must go
through `eval-leakage-check`.

## 6. Acceptance criteria

1. Every new model in `index/schema.py` and `retrieve/schema.py` round-trips `model_dump_json()` ->
   `model_validate_json()` to an equal object.
2. `get_embedder` raises `ValueError` on an unknown type. `HashEmbedder` is deterministic (two
   same-seed instances give bit-identical vectors for one text), returns `float32` shaped
   `(n, dim)` with unit-norm rows when `normalize`, and maps different texts to different vectors.
3. `SentenceTransformerEmbedder` applies prefixes: with the model object monkeypatched to capture
   inputs, `is_query=True` passes `query_prefix + text` and `is_query=False` passes
   `passage_prefix + text`; constructing the embedder alone loads no model (no download in tests).
4. `VectorStore` conformance, parameterized over both implementations (FAISS via
   `pytest.importorskip`): `add` then `search` returns the planted nearest row first; `top_k` above
   the row count returns exactly `len(store)` hits; scores are non-increasing; `save` ->
   `load_vector_store` reproduces identical `(scores, row_ids)`; `add` with a wrong dim or
   non-float32 raises `ValueError`; `get_vector_store("nope", 8)` raises `ValueError`; `len(store)`
   equals the vectors added.
5. `build_index` on a fixture `chunks.jsonl` (>= 12 chunks over >= 2 papers, hash embedder + numpy
   store) writes exactly `vectors.npy`, `chunk_meta.jsonl`, `index_manifest.json`, one JSON object
   per sidecar line in ascending `row_id` equal to the source line number; the manifest has
   `num_vectors == num_chunks == len(store)`, `num_papers` equal to the distinct `arxiv_id` count,
   and `variant_name`, `seed`, `embedder`, `index_type`, `chunks_sha256`, `fingerprint`,
   `library_versions` all populated.
6. Every `IndexedChunk` preserves its source `Chunk`'s `chunk_id`, `arxiv_id`, `page_start`,
   `page_end`, and `text` exactly — the citation contract survives indexing.
7. `build_index` is idempotent: a second call on unchanged inputs returns the same `fingerprint` and
   performs zero encode calls (counted via a monkeypatched `encode`); `force=True` re-encodes;
   editing `chunks.jsonl` changes `chunks_sha256` and triggers a rebuild.
8. `build_index` raises `FileNotFoundError` for a missing `chunks_path` and `ValueError` for an
   empty one, writing no index directory in either case.
9. `DenseRetriever.load` + `retrieve` on that index returns the exact chunk at `rank == 0` with the
   highest score in the result when the query is a chunk's own text.
10. `RetrievalResult` invariants: `len(results) <= top_k`, `rank` is `0..n-1` contiguous, `score` is
    non-increasing, `retriever == "dense"`, and `variant_name`/`embedder_model`/`index_fingerprint`
    match the manifest. `retrieve("")` and `retrieve("   ")` raise `ValueError`.
11. `DenseRetriever.load` raises `ValueError` when config's `embedder.model_name`, `dim`,
    `normalize`, or prefixes differ from the manifest's `EmbedderInfo`, and `FileNotFoundError` when
    `index_dir` has no manifest. With `max_chunks_per_paper: 1` on an index where one paper owns the
    top 3 hits, at most one hit per `arxiv_id` is returned, ranks renumbered contiguously.
12. `retrieve_batch` on n queries returns n `RetrievalResult`s, each equal (ignoring `latency_ms`)
    to the single-query `retrieve` result for the same query, with exactly one `encode` call.
13. `citations_for` returns one `Citation` per hit in rank order, with `chunk_id`, `arxiv_id`,
    `page_start`, `page_end` equal to the corresponding `RetrievedChunk`'s and transitively to the
    source `Chunk`'s — asserted end-to-end from `chunks.jsonl` through the index to the citation.
    `format_citation` renders `"arXiv:2301.12345 p.4-5"`, and `"arXiv:2301.12345 p.4"` when
    `page_start == page_end`.
14. `scripts/build_index.py` and `scripts/query.py` run end-to-end on `config/offline_smoke.yaml`
    against a fixture corpus in `tmp_path` and exit 0, with `query.py` printing one citation line
    per hit; `query.py` exits non-zero on an empty query and on a missing index directory; `--out`
    refuses a path under an `eval` directory (leakage guard, §5).
15. `load_config("baseline")` and `load_config("offline_smoke")` both succeed and their key sets
    match §4 exactly (config-completeness test, so an ablation copy cannot silently drop a key), and
    neither contains a `generation` block.
16. No test downloads model weights, constructs a real `SentenceTransformer`, or makes a network
    request (enforced by monkeypatching the embedder's model loader); `.gitignore` ignores
    `data/index/`; `numpy` appears in `pyproject.toml` `dependencies`; and `src/docsense/generate/`
    contains only its `__init__.py`.
17. `check` skill (pytest -q + ruff check .) passes with these tests added.

## 7. Decisions (formerly open questions, settled)

Nothing here is open; the reviewer should treat all five as fixed.

1. **Generation deferred — settled by the human.** Phase 2 ships retrieval only: no LLM client, no
   prompt file, no `Answer` record, no `generation.*` config block, and `src/docsense/generate/`
   stays an empty package. Rationale: Phase 3's retrieval metrics should not be blocked on choosing
   and funding a provider. **Deferred, not dropped** — §2's "Deferred: generation" subsection holds
   the contract (numbered context blocks, prompt in `config/prompts/`, model emits `[n]` markers
   only, citations resolved through `citation_for`) for whichever later phase picks it up, which
   must also settle the per-run API cost ceiling that this phase no longer needs to answer. Expected
   consequence: Phase 3 measures Recall@k / nDCG@k / MRR and cannot score answer quality yet.
2. **Embedder — confirmed.** `BAAI/bge-small-en-v1.5` (384d) as both the baseline retriever and
   Phase 4's fine-tune base. `query_prefix` stays the bge instruction string; a different model
   would require changing it.
3. **Device — confirmed as auto-detect.** `embedder.device: null`. Accepted consequence: on a
   machine with MPS or CUDA the chosen device shows up in the index manifest's `library_versions`
   context but not as a pinned setting, and MPS float results differ slightly from CPU, so a
   reported number is reproducible only on the same device. If Phase 7 needs cross-machine
   bit-identity, pin `device: "cpu"` in the ablation configs then.
4. **`retrieval.top_k: 5` — accepted as the baseline** (~1.5k tokens of context when generation
   arrives). Phase 7 ablates around it (3/5/10) by config copy.
5. **Corpus scope and dependencies — accepted as proposed.** `build_index` indexes all of
   `chunks.jsonl` with no held-out split, so Phase 3 may draw gold questions from any paper in the
   committed manifest; and `numpy` becomes a declared dependency in `pyproject.toml` rather than a
   transitive one, since this phase imports it directly.
