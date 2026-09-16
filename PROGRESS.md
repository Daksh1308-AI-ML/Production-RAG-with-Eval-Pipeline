# Progress Report

**Project:** Production RAG with Eval Pipeline — SEC 10-K Filings
**Last updated:** 2026-09-16 (finalized)

## Completed

### Design & Planning
- `PROJECT_REQUIREMENTS.md` — project scope, corpus (AAPL, MSFT, GOOGL, AMZN, NVDA × 3 years), success criteria
- `SYSTEM_ARCHITECTURE.md` — hybrid retrieval (BM25 + dense), reranking, RAGAS eval, Streamlit, Langfuse
- `TECHNICAL_DESIGN.md` — component specs, chunking (1000/200), retrieval tuning (BM25 0.4 / Dense 0.6, k 20 → 5)
- `RAG_EVALUATION.md` — metrics (faithfulness, answer_relevancy, context_precision, context_recall)
- `ROADMAP.md` — phased delivery plan
- `README.md` — added (overview, setup, config, quick start)

### Environment & Project Setup
- Project directory structure created: `data/sec_filings/{aapl,msft,googl,amzn,nvda}`, `data/processed`, `data/evaluation`, `src`, `app`, `tests`, `notebooks`, `docker`
- Python venv created and dependencies installed
- `.env` created
- Git repository initialized
- `requirements.txt` updated: pinned `ragas==0.4.3`, added `pytest`

### Source Code (all 13 modules import cleanly)
| Module | Purpose |
|--------|---------|
| `src/config.py` | Central config (Ollama, Qdrant, chunking, retrieval, eval, monitoring) |
| `src/ingest.py` | SEC EDGAR 10-K ingestion |
| `src/chunker.py` | Document chunking (1000 tokens / 200 overlap) |
| `src/embed.py` | Ollama `nomic-embed-text` embeddings (768-dim) |
| `src/store.py` | Qdrant vector store (create/upsert/search/retriever) |
| `src/retrieve.py` | Hybrid retrieval (BM25 0.4 + dense 0.6, ensemble) |
| `src/rerank.py` | BGE reranker (`BAAI/bge-reranker-base`) |
| `src/query_rewrite.py` | Query rewriting/preprocessing |
| `src/generate.py` | Ollama `qwen2.5:7b` answer generation |
| `src/rag.py` | End-to-end RAG pipeline orchestration |
| `src/eval.py` | RAGAS evaluation pipeline |
| `src/monitoring.py` | Langfuse tracing (optional) |
| `src/utils.py` | Helpers |

- `src/eval.py` rewritten for RAGAS 0.4.3 API (`SingleTurnSample`, `EvaluationDataset`, `evaluate(..., raise_exceptions=True)`)
- **Judge now configurable** (`src/config.py` + `src/eval.py`): `JUDGE_API_KEY` empty → local Ollama `qwen2.5:7b` judge (unchanged fallback); key set → `ChatOpenAI` against any OpenAI-compatible API (OpenRouter/Groq/Cerebras/etc.) with enforced JSON mode (`response_format: {"type":"json_object"}`), fixing RAGAS's strict `model_validate_json` failures on local qwen's prose/fenced output. Embeddings stay local (Ollama `nomic-embed-text`). `langchain-openai` added to `requirements.txt`. Verified: `ChatOpenAI` branch + `ChatOllama` fallback both construct; `pytest` 2 passed / 3 skipped.
- **RAGAS integration tested end-to-end** (smoke): `--limit 2 --strategies baseline --metrics faithfulness` → **baseline faithfulness 0.67** (0.71 on a 3-sample run), full per-sample details exported. Two fixes along the way:
  - `openrouter/free` router **randomizes the model per call**, so ragas' strict two-step schema (string statements, then verdicts) got flaky/non-JSON output (e.g. a `User Safety: safe` preamble) → **pin one model**: pinned `JUDGE_MODEL=nvidia/nemotron-3-super-120b-a12b:free` (proven) — as of 2026-09 that listing stopped serving (404), swapped to `nvidia/nemotron-3.5-lightning:free`, the current reliable free pick with structured-output support.
  - `src/eval.py` detail export used non-existent `EvaluationResult.save_to_json` → replaced with `details.to_pandas().to_json(...)` (RAGAS 0.4.3 API).
- `scripts/build_eval_dataset.py` — eval dataset builder with **103 QA pairs** (AAPL/MSFT/GOOGL/AMZN/NVDA + 4 cross-company comparatives), ground truths and `reference_contexts` lifted verbatim from the filings. Runner asserts unique questions and that every reference context is a whitespace-collapsed substring of `data/processed/filings.json`.
  - Dataset mined in **5 parallel sub-agents** (15 pairs per ticker), merged with the original 30, then deduped (dropped 2 near-duplicate questions) → 103.
  - Distribution: factual 67 / analytical 21 / comparative 12 / summary 3; easy 25 / medium 51 / hard 27; all 169 reference contexts verified verbatim.
  - 8 verbatim-context fixes: AAPL Services row uses `Services (1) 109,158 %...`, AAPL risk text needed the curly-apostrophe `\u2019` (raw filings use U+2019), GOOGL intl-lines add `in 2025`, NVDA segment rows are `Compute & Networking $ 193,479 $ 116,193 $ 77,286 %` (no `Total Graphics Revenue` heading), AMZN AWS row includes the full `Operating expenses` column, AMZN Anthropic sentence contains an in-text `Table of Contents` page marker, NVDA R&D third value is the `$ Change` column (not FY2024).
- `pytest` fully green later on (**5/5 passed** — integration tests run once Qdrant/Ollama are up).
- `src/ingest.py` fixed for current `sec-edgar-downloader` API (`email_address` kwarg); parses `full-submission.txt` files (the downloader no longer writes `.htm`), strips the SEC header from chunk text, and extracts ticker + filing date from the EDGAR header block
- `src/store.py` `upsert_documents` now embeds in batches (200/request) instead of per-chunk HTTP calls

### Infrastructure
- Docker Desktop running (daemon accessible)
- Qdrant server `v1.19.0` running on `localhost:6333`
- `docker/docker-compose.yml` cleaned: removed obsolete `version:`, pinned server `v1.19.0`
- Ollama serving `qwen2.5:7b` (LLM + judge), `nomic-embed-text` (embeddings); judge may alternatively use a free hosted API via `JUDGE_MODEL`/`JUDGE_BASE_URL`/`JUDGE_API_KEY`

### Data Pipeline (complete)
- Downloaded 15 SEC 10-K filings (5 tickers × 3 years) to `data/sec_filings/sec-edgar-filings/<TICKER>/10-K/`
- Parsed + cleaned all 15 filings; cached to `data/processed/filings.json` (86s)
- Chunked to **39,565 chunks** (1000 chars / 200 overlap)
- Indexed into Qdrant `sec_filings` collection — **39,565 points** verified via `points_count`
  - `scripts/ingest_index.py` (reproducible loader: load → chunk → recreate collection → upsert with progress prints)
  - Full run: 756s (~52 chunks/s on CPU with batched Ollama embeddings)

### Week 1: Dense Retrieval + Basic RAG Chain (validated)
- `scripts/validate_retrieval.py` — dense retrieval smoke test on 6 sample queries + `ticker=AAPL` metadata filter: **PASS**
  - Per-query latency 68–102ms (first query 2.2s, cold embed); scores surfaced on results
  - Known dense-only gap: "Amazon operating segments" pulled MSFT/AAPL rows on top (expected — hybrid + rerank in Week 2)
- `scripts/basic_rag_demo.py` — dense retrieve → `qwen2.5:7b` generation with citations: **working** (Apple FY24 net sales $391.0B, MSFT cloud revenue, NVDA data center revenue all answered with correct figures + source refs)
- `src/store.py` — migrated Qdrant `search()` → `query_points()` (v1.19 API removed `client.search`); scores attached to result metadata
- `tests/test_retrieve.py` — integration tests (skip when Qdrant/Ollama down): collection count 39565, k-doc search with scores, metadata filter. **5/5 tests pass**

### Week 2: Hybrid Retrieval + Reranking (validated)
- `scripts/validate_hybrid.py` — BM25 + dense RRF + `bge-reranker-base` on 6 sample queries: **PASS** (all ticker checks clean, `RESULT: PASS`)
  - Full run ~2 min on CPU: BM25 60–130ms, dense 75ms–2.2s, hybrid 130–190ms, rerank 2.0–4.8s
  - Fixed first-run model download stall: pre-fetch only the files the app needs via `snapshot_download(allow_patterns=...)` — a full fetch of `ms-marco-MiniLM-L6-v2` pulls ~865 MB of redundant ONNX/OpenVINO/Flax/PyTorch copies and deadlocks (CloseWait socket) on slow Hugging Face connections
  - Verified cached loads: `bge-reranker-base` ~12–15s (~1.1 GB), `cross-encoder/ms-marco-MiniLM-L6-v2` ~11s; pre-download recipe documented in `README.md` §5

### Week 2: Advanced Retrieval (validated)
- `src/retrieve.py` — replaced `EnsembleRetriever` (silently dropped results below `k`, ignored `filter_metadata`) with 8-line manual weighted RRF; added `retrieve_multi()` for query-expansion fusion
- `src/rag.py` — query rewriting now feeds `retrieve_multi` (rewritten queries + original fused via RRF) instead of retrieving on the original query only
- `scripts/validate_hybrid.py` — now covers the rewrite path: 6 ticker queries + 1 ambiguous rewrite query; `Reranker(model_name=...)` override
- Full Week 2 run **PASS**: hybrid 115–183ms (target <300ms), exactly 20 docs returned (old bug: 13–18); rewrite via Ollama 13.7s; rerank 0.34–0.58s (`ms-marco-MiniLM-L6-v2`) / 2–4.8s (`bge-reranker-base`)
- Rerank compression = `Reranker` top-20 → top-5; separate LangChain compression wrapper skipped (YAGNI)

### Verification
- All 13 source modules import without errors
- `tests/` pass (5/5): `test_config_initialization`, `test_ingestor_initialization`, `test_collection_populated`, `test_dense_search_returns_k_docs`, `test_metadata_filter` (pytest 9.1.1)
- `VectorStore` connects to Qdrant; collection exists without warnings
- Corpus indexing verified end-to-end (39565 chunks == 39565 points)
- Dense retrieval smoke test PASS; basic RAG chain (retrieve → qwen2.5:7b → citations) verified on 3 questions

## Pending / Next Steps

- [x] **Week 3:** RAGAS evaluation — setup + smoke test done (see Completed); full run across all 4 strategies pending
- [x] Run RAGAS smoke evaluation; `data/evaluation/eval_dataset.json` generated from `scripts/build_eval_dataset.py` (30 QA pairs)
- [ ] Run full RAGAS evaluation across `baseline,hybrid,rerank,full` strategies and record results (**Day 20-21, deferred to final testing per owner decision — harness + 103-pair dataset ready**)
- [x] Build Streamlit UI (`app/streamlit_app.py`) — **DONE 2026-09-14**: chat UI + strategy selector + citations + latency; smoke-tested end-to-end (baseline query, 20 sources)
- [x] Enable Langfuse monitoring — **wired but disabled** (v4 SDK; needs `LANGFUSE_*` keys)
- [x] Dockerfile + docker-compose app service — **DONE 2026-09-14** (build+run test pending)
- [x] README metrics/results section — covered by RAG_EVALUATION.md + A/B framework
- [x] Architecture diagrams — **DONE 2026-09-16**: 3 SVGs in `assets/images/` embedded in README
- [ ] **Day 7-21 A/B deferred** — eval Q/A at end: full RAGAS run, failure analysis, `notebooks/ab_test_analysis.ipynb`

## Phase 2 (complete, 2026-09-16)

Built and verified — commits `934605d` + follow-ups.

1. **Multi-tenant (payload-filter):** single Qdrant collection `sec_filings`, chunks tagged with `tenant_id` payload; retrieval filters by tenant. New `TENANT_ID` env (default `"default"`). No collection-per-tenant.
2. **Real-time ingestion:** `scripts/ingest_one.py` ingests individual filings incrementally (parse → chunk → upsert, no collection recreation). `src/store.py` incremental upsert deletes existing points for a source, keyed by hash of source + chunk_index. `scripts/ingest_index.py` gains a `--tenant` flag.
3. **Advanced caching (semantic cache):** `src/cache.py` `SemanticCache` in a Qdrant `semantic_cache` collection, served from cache before retrieval when top-1 cosine similarity ≥ `CACHE_THRESHOLD` (0.92); written after generation; bypassed for `baseline`; wiped on re-ingest.
4. **API gateway:** FastAPI `app/api.py` — `POST /query`, `POST /ingest`, `GET /health`; auth via `X-API-Key` against comma-separated `API_KEYS` env; pipeline cached per strategy; `api` service in docker-compose (port 8000).

New files: `src/cache.py`, `app/api.py`, `scripts/ingest_one.py`. New env vars: `TENANT_ID`, `CACHE_ENABLED`, `CACHE_THRESHOLD`, `API_KEYS`.

## Phase 3 (implemented 2026-09-16)

Built and wired in. Verified: module unit checks pass, pytest 5/5, end-to-end query smoke (cache hit + injection blocked) OK. Self-RAG refuse/expansion paths exist but were NOT run against the live LLM/reranker (deferred). No score files in `results_ab` yet (only `responses_*`), so `ab_report.py` shows "no scored samples" until a full `run_ab` is executed.

1. **Self-RAG (adaptive retrieval)** — `src/selfrag.py` `SelfRag`. Reranker cross-encoder score attached to docs (`rerank_score`). Best score below `SELF_RAG_MIN_CONFIDENCE` (0.3) → re-retrieve with wider `SELF_RAG_EXPAND_K` (40) + re-rerank once; still below `SELF_RAG_REFUSE_BELOW` (0.15) → refuse to answer. Active for `full`/`rerank` strategies in `src/rag.py`.
2. **Guardrails** — `src/guardrails.py` `Guardrails`. Input guard regex-blocks prompt-injection patterns, PII (SSN, cards, emails, phones), off-topic keywords → canned refusal. Output guard refuses leaked PII / botched refusals. Wired into `src/rag.py`. `GUARDRAILS_ENABLED` (default true).
3. **Analytics dashboard** — `app/streamlit_app.py` sidebar `View` toggle `Chat | Analytics`: Qdrant chunk count (`sec_filings`), semantic-cache entry count, session cache hit rate (tracked in `src/cache.py`), session latency, A/B results (`ab_report.md` → `summary.json` → run hint).
4. **A/B testing framework** — `scripts/ab_report.py` reads `results_ab/scores_{strategy}_{metric}.json`, computes per-strategy/per-metric means, picks best strategy per metric, writes `ab_report.md`; `--no-write` prints only.

New files: `src/selfrag.py`, `src/guardrails.py`, `scripts/ab_report.py`. New env vars: `SELF_RAG_ENABLED`, `SELF_RAG_MIN_CONFIDENCE`, `SELF_RAG_REFUSE_BELOW`, `SELF_RAG_EXPAND_K`, `GUARDRAILS_ENABLED`.

## Known Notes
- **`src/eval.py` `--dataset` bug FIXED** (2026-09-14): `main()` validated the flag but `load_eval_dataset()` always read `config.eval.eval_dataset_path` — CLI path was ignored (first A/B launch silently evaluated all 103 pairs instead of the 8-pair subset). Fix: `load_eval_dataset(dataset_path=None)`; `main()` passes `args.dataset`. Verified: loads 8 samples from `ab_stratified.json`, pytest 5/5.
- `scripts/make_ab_subset.py` (untracked) + `data/evaluation/ab_stratified.json` (gitignored) — stratified 8-pair A/B subset, kept for the deferred final testing phase.
- `data/evaluation/results_ab/` removed; background eval PID 25372 (launched on the unfixed bug) exited before the fix.

## Finalization (2026-09-16)

- Framework complete across 3 phases: ingestion → hybrid retrieval + rerank → RAGAS eval, then Phase 2 (API gateway, semantic cache, multi-tenant, incremental ingestion) and Phase 3 (Self-RAG, guardrails, analytics dashboard, A/B report).
- README finalized with architecture images (`assets/images/*.svg`) and phase-complete status.
- Repository pushed to GitHub (final version).