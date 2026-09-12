# Production RAG with Eval Pipeline

A production-grade **Retrieval-Augmented Generation (RAG)** system that answers natural-language questions about **SEC 10-K filings** — with hybrid retrieval, reranking, and a RAGAS-based evaluation pipeline. Designed as a portfolio-ready project that separates quality RAG from naive vector search.

## Features

- **SEC 10-K ingestion** — downloads and parses 15 filings (AAPL, MSFT, GOOGL, AMZN, NVDA × 3 years) from SEC EDGAR
- **Hybrid retrieval** — BM25 (0.4) + dense vector search (0.6) fused via Reciprocal Rank Fusion (top-20 candidates)
- **Reranking** — `BAAI/bge-reranker-base` cross-encoder re-ranks top-20 → top-5
- **Answer generation** — local `qwen2.5:7b` via Ollama, with source attribution and refusal when context is insufficient
- **RAGAS evaluation** — faithfulness, answer relevancy, context precision/recall, judged locally by Ollama
- **100% local & open source** — everything runs on your machine via Ollama + Docker; no cloud AI APIs
- **Streamlit UI** — chat interface with citations
- **Langfuse monitoring** — optional tracing (latency, cost, errors)

## Architecture

```
User Query → Query Rewrite → Hybrid Retrieval (BM25 + Dense, RRF)
          → Rerank (top-20 → top-5) → Generate (qwen2.5:7b) → Answer + Citations
```

Stores: **Qdrant** (vector DB), **Ollama** (embeddings `nomic-embed-text`, LLM), **SEC EDGAR** (source data). See [`SYSTEM_ARCHITECTURE.md`](SYSTEM_ARCHITECTURE.md) for full diagrams, [`TECHNICAL_DESIGN.md`](TECHNICAL_DESIGN.md) for component specs.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama · `qwen2.5:7b` |
| Embeddings | Ollama · `nomic-embed-text` (768-dim) |
| Vector DB | Qdrant |
| Orchestration | LangChain |
| Retrieval | BM25 + dense (RRF ensemble) |
| Reranking | `BAAI/bge-reranker-base` |
| Evaluation | RAGAS |
| UI | Streamlit |
| Monitoring | Langfuse (optional) |

## Prerequisites

- Python 3.11+
- Docker Desktop (for Qdrant)
- Ollama installed with models pulled:
  - `ollama pull qwen2.5:7b`
  - `ollama pull nomic-embed-text`
- 16 GB+ RAM recommended (LLM runs locally)

## Quick Start

### 1. Start services

```bash
docker compose -f docker/docker-compose.yml up -d qdrant
# Ollama should be running too (native install or docker compose service, uncomment if used)
```

Verify:
```bash
curl http://localhost:6333/collections   # Qdrant
curl http://localhost:11434/api/tags     # Ollama
```

### 2. Install Python environment

```bash
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS/Linux
pip install -r requirements.txt
```

### 3. Configure `.env`

Create `.env` from the variables documented in [Configuration](#configuration) below (see `src/config.py`). `.env` is git-ignored — never commit secrets.

### 4. Ingest the corpus

```bash
python -c "from src.ingest import SECIngestor; print(SECIngestor().download_all())"
python -c "from src.chunker import DocumentChunker; from src.store import VectorStore; from src.config import config; import json; filings=json.load(open(config.processed_dir/'filings.json')); chunks=DocumentChunker().chunk_documents(filings); vs=VectorStore(); vs.create_collection(recreate=True); print('upserted', vs.upsert_documents(chunks))"
```

The first ingestion embeds ~40,000 chunks with `nomic-embed-text` on CPU — allow ~1–3 hours. Files land in `data/sec_filings/` and `data/processed/`.

### 5. Run the app

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
├── app/                 # Streamlit UI
├── docker/              # docker-compose (Qdrant, Ollama)
├── data/
│   ├── sec_filings/     # raw 10-K downloads (git-ignored)
│   ├── processed/       # parsed filings, chunk cache (git-ignored)
│   └── evaluation/      # eval datasets / results
├── src/
│   ├── config.py        # central configuration
│   ├── ingest.py        # SEC EDGAR 10-K download + parse
│   ├── chunker.py       # text chunking (1000 / 200 overlap)
│   ├── embed.py         # Ollama embeddings
│   ├── store.py         # Qdrant vector store
│   ├── retrieve.py      # hybrid retrieval (BM25 + dense, RRF)
│   ├── rerank.py        # bge-reranker cross-encoder
│   ├── query_rewrite.py # query expansion
│   ├── generate.py      # answer generation
│   ├── rag.py           # end-to-end RAG pipeline
│   ├── eval.py          # RAGAS evaluation
│   └── monitoring.py    # Langfuse tracing
├── tests/               # pytest suite
└── notebooks/           # analysis notebooks
```

## Configuration

All settings live in `src/config.py` and read from env vars (loaded from `.env`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_LLM_MODEL` | `qwen2.5:7b` | Generator + judge |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant connection |
| `QDRANT_COLLECTION` | `sec_filings` | Collection name |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Chunking |
| `BM25_WEIGHT` / `DENSE_WEIGHT` | `0.4` / `0.6` | Retrieval weights |
| `INITIAL_K` / `FINAL_K` | `20` / `5` | Retrieve → rerank |
| `SEC_COMPANY_NAME` / `SEC_EMAIL` | — | SEC EDGAR user agent |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | — | Monitoring (optional) |

## Testing

```bash
python -m pytest tests/
```

## Documentation

- [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md) — scope, requirements, success criteria
- [`SYSTEM_ARCHITECTURE.md`](SYSTEM_ARCHITECTURE.md) — architecture & data flow
- [`TECHNICAL_DESIGN.md`](TECHNICAL_DESIGN.md) — component specifications
- [`RAG_EVALUATION.md`](RAG_EVALUATION.md) — metrics & methodology
- [`ROADMAP.md`](ROADMAP.md) — phased delivery plan
- [`PROGRESS.md`](PROGRESS.md) — current status / next steps