# Progress Report

**Project:** Production RAG with Eval Pipeline — SEC 10-K Filings
**Last updated:** 2026-09-13

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

- [ ] **Week 3:** RAGAS evaluation (Week 2 hybrid retrieval + reranking complete — see Completed)
- [ ] Run RAGAS evaluation, generate `data/evaluation/eval_dataset.json`
- [ ] Build Streamlit UI (`app/streamlit_app.py` placeholder only)
- [ ] Optional: enable Langfuse monitoring
- [ ] Initial git commit

## Known Notes
- `langchain-community` deprecation warning observed (expected, non-blocking)
- `langgraph`, `langchain-classic`, `langchain-openai` uninstalled to resolve conflicts; may be re-triggered by ragas installs (not currently installed)
- `data/.env.example` no longer present after final `.env` copy; README documents the config vars instead