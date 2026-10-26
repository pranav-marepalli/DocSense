# DocSense

Multimodal retrieval-augmented QA over arXiv ML papers (text, tables, figures) with page-level citations, a hand-labeled evaluation set, and an ablation study. This is a portfolio project for ML internship applications; the evaluation and ablation results are the most important deliverable.

## Stack
- Python 3.11, pytest, ruff
- rank_bm25, sentence-transformers, faiss (or a lightweight vector store), PyMuPDF/Docling, FastAPI, Docker
- No LangChain or LlamaIndex for core logic

## Conventions
- All settings in config/*.yaml; each ablation variant is a config file, not a code change
- Set seeds; log model and data versions with every eval run
- Layout: src/docsense/{ingest,index,retrieve,multimodal,generate,router,train,eval,serve}, scripts/, tests/, docs/design/
- Design docs live in docs/design/phase-N.md and are the source of truth for each phase

## Hard rules
- Never create, edit, or fabricate evaluation labels or results. The human labels the gold set.
- Synthetic training data must never include passages from the eval set; use the `eval-leakage-check` skill (wraps `scripts/check_eval_leakage.py`).
- Use the `check` skill (pytest -q + ruff check .) before declaring any work done.

## Subagents and skills
- Roles are split across subagents in `.claude/agents/`: `architect` (design only), `implementer` (code + tests), `reviewer` (independent QA, never edits). Drive a phase with `/phase <N>`, which orchestrates design -> implement -> review and pauses for human approval after the design step.
- Repeated mechanical checks are skills in `.claude/skills/`, not re-derived each time: `check` (test+lint gate) and `eval-leakage-check` (n-gram overlap between synthetic training data and the eval set).

## Phases
1. Ingestion  2. Baseline RAG  3. Eval harness  4. Hybrid + reranker + fine-tuned embedder
5. Multimodal + query router  6. FastAPI serving + logging  7. Full ablation + README
