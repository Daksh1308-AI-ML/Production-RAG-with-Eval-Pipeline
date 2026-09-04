# Progress Report

**Project:** Production RAG with Eval Pipeline — SEC 10-K Filings
**Last updated:** 2026-09-04

## Completed

### Design & Planning
- `PROJECT_REQUIREMENTS.md` — project scope, corpus (AAPL, MSFT, GOOGL, AMZN, NVDA × 3 years), success criteria
- `SYSTEM_ARCHITECTURE.md` — hybrid retrieval (BM25 + dense), reranking, RAGAS eval, Streamlit, Langfuse
- `TECHNICAL_DESIGN.md` — component specs, chunking (1000/200), retrieval tuning (BM25 0.4 / Dense 0.6, k 20 → 5)
- `RAG_EVALUATION.md` — metrics (faithfulness, answer_relevancy, context_precision, context_recall)
- `ROADMAP.md` — phased delivery plan

### Environment & Project Setup
- Project directory structure created: `data/sec_filings/{aapl,msft,googl,amzn,nvda}`, `data/processed`, `data/evaluation`, `src`, `app`, `tests`, `notebooks`, `docker`
- Python venv created and dependencies installed
- `.env` created from `.env.example`
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

- `src/eval.py` rewritten for RAGAS 0.4.3 API (`SingleTurnSample`, `EvaluationDataset`, `evaluate(..., raise_exceptions=True)`) — previously blocked on `Dataset.from_dict` import error

### Infrastructure
- Docker Desktop running (daemon accessible)
- Qdrant server `v1.19.0` running on `localhost:6333` (matched to installed `qdrant-client 1.19.0`, resolving the version-mismatch warning)
- `docker/docker-compose.yml` cleaned: removed obsolete `version:` attribute, pinned server `v1.19.0`
- Ollama serving `qwen2.5:7b` (LLM + judge) and `nomic-embed-text` (embeddings)

### Verification
- All 13 source modules import without errors
- `tests/` pass: `test_config_initialization`, `test_ingestor_initialization` (pytest 9.1.1)
- `VectorStore` connects to Qdrant and created the `sec_filings` collection successfully (no warnings)

## Pending / Next Steps

- [ ] Ingest SEC 10-K filings (15 filings: 5 tickers × 3 years) — `src/ingest.py` ready
- [ ] Chunk + embed + upsert corpus into Qdrant
- [ ] Validate hybrid retrieval & reranking
- [ ] Run RAGAS evaluation, generate `data/evaluation/eval_dataset.json`
- [ ] Build Streamlit UI (`app/streamlit_app.py` placeholder only)
- [ ] Optional: enable Langfuse monitoring
- [ ] Initial git commit

## Known Notes
- `langchain-community` deprecation warning observed (expected, non-blocking)
- `langgraph`, `langchain-classic`, `langchain-openai` uninstalled to resolve conflicts; may be re-triggered by ragas installs (not currently installed)