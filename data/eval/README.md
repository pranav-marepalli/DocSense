# The DocSense gold set

`gold_set.jsonl` is the project's one hand-labeled evaluation set. It ships
**empty** (zero bytes). Per `CLAUDE.md`'s hard rule -- "Never create, edit, or
fabricate evaluation labels or results. The human labels the gold set." -- no
code in `src/docsense/` writes to this file, infers a label from a retrieval
result, or generates a synthetic query/label pair. The only writer is
`scripts/label_eval.py`, and only through a human keystroke per candidate.

See `docs/design/phase-3.md` for the full design. This file is the
quick-reference for the human doing the labeling.

## Workflow

1. Build the index if you have not already:
   ```
   python scripts/build_index.py --config baseline
   ```
2. Skim some papers to write queries about:
   ```
   python scripts/label_eval.py --config eval --suggest-papers 20
   ```
   This prints `arxiv_id`, `query_category`, and `title` for a seeded sample
   of `data/manifest.jsonl` rows. It writes nothing.
3. Write questions into `data/eval/queries.txt`, one per line (`#`-prefixed
   lines are comments). Phrase them as a real user would ask -- not copied
   verbatim from a paper's sentences, which would make retrieval trivially
   lexical. Aim for a spread across `query_type` (including `table` and
   `figure` queries, which Phase 5 needs) and across the four
   `fetch.categories`.
4. Judge each query's retrieved pool:
   ```
   python scripts/label_eval.py --config eval --labeled-by "<your name>"
   ```
   For each query not already in the gold set, this retrieves
   `labeling.pool_k` candidates (default 20, deliberately deeper than the
   `retrieval.top_k`/`eval.retrieval_top_k` actually scored) and shows each
   one's citation and a text snippet. You respond with:
   - `0` / `1` / `2` -- grade the candidate (0 = judged not relevant, 2 = highly
     relevant).
   - `s` -- skip this candidate without grading it.
   - `a` -- add a span by hand, for a passage you know is relevant that the
     pool missed, e.g. `2301.00001:4-6 2` (`<arxiv_id>:<page_start>-<page_end>
     <relevance>`).
   - `u` -- mark the whole query unanswerable (no relevant passage exists in
     the corpus).
   - `n` -- stop grading this query's remaining candidates and finalize it.
   - `q` -- save and quit the session.

   Blank or unrecognized input re-prompts; there is no default grade and no
   "accept all" shortcut. After a query is finalized you are also prompted for
   its `query_type` and `difficulty`.

   `label_eval.py` never rewrites or deletes an existing line -- it only
   appends -- so re-labeling an example later is a deliberate, visible file
   edit.
5. Get the first real baseline numbers and commit the labels:
   ```
   python scripts/eval.py --config eval
   git add data/eval/gold_set.jsonl data/eval/queries.txt
   ```

## File format

One `GoldExample` per line (newline-delimited JSON), append-only:

| Field | Meaning |
|---|---|
| `query_id` | `"q001"`-style, unique within the file |
| `query` | The question text |
| `query_type` | One of `factoid`, `definition`, `method`, `comparison`, `numeric_result`, `table`, `figure`, `multi_hop` |
| `difficulty` | `easy` / `medium` / `hard` |
| `unanswerable` | `true` means no relevant passage exists; no span may have `relevance >= 1` |
| `spans` | List of `LabeledSpan`: `arxiv_id`, `page_start`, `page_end` (1-indexed, inclusive), `relevance` (0-2), `note`, `source_chunk_id` |
| `tags` | Free-form labels, e.g. for later slicing |
| `labeled_by` | Who made the judgment |
| `labeled_at` | ISO 8601 UTC timestamp |
| `pooled_from` | Retriever names whose candidates the labeler saw (`[]` = fully manual, e.g. a hand-added span) |
| `pool_k` | Pool depth reviewed, for auditing pooling bias |

Each `LabeledSpan` is keyed on a **page span**, not a `chunk_id`. `chunk_id`s
encode the chunking config (`chunk_size_tokens`), so a chunk_id-keyed label
would be invalidated by exactly the re-chunking ablations Phase 7 runs; page
spans are chunking-invariant and already the project's citation unit.
`source_chunk_id` is kept on a span only as provenance (the chunk the labeler
was shown) and is **never scored**.

## Important rules

- **`relevance == 0` is a recorded judgment, not a training negative.** It
  means a human looked at that candidate and said it was not relevant to the
  query. Mining relevance-0 spans as Phase 4 hard negatives would train on
  the eval set -- don't do it.
- **nDCG here is first-credit-only, a documented deviation from textbook
  nDCG** (`docsense.eval.relevance.credit_hits`): a labeled span pays out its
  gain at most once, at the highest rank that first covers it. Textbook nDCG
  gives every relevant item its gain independently, so three overlapping
  chunks of the same passage would triple-count and a retriever that returns
  near-duplicates of one hit would score close to 1.0 -- a real failure mode
  for a chunked corpus with overlapping windows. `ideal_gains` is the span
  relevances sorted descending, so `ndcg@k == 1.0` means "the top ranks cover
  distinct relevant spans, best first", not "every relevant chunk was
  returned".
- **Pooling bias.** Labels are drawn from `retriever_config`'s pool at
  `pool_k` depth, so Recall is biased upward toward what that retriever can
  find; `pooled_from` and `pool_k` are recorded per example precisely so this
  is auditable. The `a` manual-add command is the mitigation for a known-
  relevant passage the pool missed. Phase 4 re-pools with the hybrid
  retriever and re-labels the union (the standard TREC remedy) rather than
  discovering this bias late.
- **Labeling must finish, and not be revised, before comparing variants.**
  `gold_sha256` is recorded in every `EvalReport` precisely so a post-hoc
  label edit is visible in both the git history and the report it produced.

## Recommended effort

At least 50 queries for headline numbers; 80-120 if you want slice
breakdowns (`query_type`, `difficulty`) to mean anything. Aim for 2-4
relevant spans per query and roughly 10% deliberately `unanswerable`. Budget
roughly 3-6 minutes per query (writing the question plus judging a 20-item
pool). At n=50 the standard error on Recall@5 is roughly 0.07, so only
differences above about 0.10 are statistically defensible -- Phase 7 should
report a paired bootstrap or sign test over per-query scores rather than two
bare means.
