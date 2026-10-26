# DocSense

Multimodal retrieval-augmented QA over arXiv ML papers (text, tables, figures) with
page-level citations, a hand-labeled evaluation set, and an ablation study. This is a
portfolio project for ML internship applications; the evaluation and ablation results
are the most important deliverable.

This repo is being built in phases (ingestion, baseline RAG, eval harness, hybrid
retrieval + reranker + fine-tuned embedder, multimodal + query router, FastAPI
serving, and a full ablation study). See `docs/design/phase-N.md` for the design doc
of each phase as it lands; this README currently documents what Phase 1 (ingestion)
delivers.

## Rebuilding the corpus

arXiv papers are not uniformly licensed for redistribution, so the downloaded PDFs
(`data/raw/`) and everything extracted from them (`data/parsed/`) are gitignored.
What *is* committed is `data/manifest.jsonl`: one row per paper
(`arxiv_id`, `version`, `sha256`, `pdf_path`, ...), which is enough to reproduce the
corpus exactly. To rebuild it from a fresh clone:

```bash
git clone <this repo>
cd mlproj

python scripts/fetch_arxiv.py --config ingest
# Re-fetches exactly the papers named in data/manifest.jsonl, at the version each
# row recorded, verifying sha256 per file -- a mismatch is reported as a per-paper
# failure rather than a silent substitution (see docs/design/phase-1.md §3).

python scripts/ingest.py --config ingest
# Extracts those PDFs into page-aware text, tables, figure crops, and chunks under
# data/parsed/ (gitignored, regenerated from the PDFs every time).
```

`scripts/fetch_arxiv.py --config ingest --dry-run` lists what the first command
would (re-)download without writing anything, which is useful for checking the
rebuild plan before running it for real.

**Current manifest size.** `data/manifest.jsonl` currently holds a 4-paper
smoke-test subset -- one paper per fetch category (`cs.CL`, `cs.LG`, `cs.CV`,
`cs.IR`) -- used to exercise the ingestion pipeline end-to-end. It is **not** yet
the full corpus targeted by `config/ingest.yaml`'s `fetch.max_papers: 300`. The full
fetch (stratified evenly across the four categories, per `docs/design/phase-1.md`)
will run later, before the Phase 3 gold-set evaluation labeling, so that the eval
set is drawn from a frozen, already-committed manifest.

## Development

```bash
pip install -e .
pytest -q
ruff check .
```
