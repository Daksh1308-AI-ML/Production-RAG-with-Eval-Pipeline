# Project Requirements

## Problem Statement

Companies lose millions querying internal documents with inaccurate AI. Simple vector search returns irrelevant chunks and hallucinated answers. This project builds a production-grade RAG system with evaluation rigor that separates it from 90% of candidates.

## Business Requirements

### Core Functionality
1. **Document Querying**: Users can ask natural language questions about SEC 10-K filings
2. **Source Citations**: Every answer includes references to specific document sections
3. **Hybrid Retrieval**: Combines semantic (vector) and keyword (BM25) search
4. **Reranking**: Post-retrieval precision improvement via cross-encoder
5. **Query Rewriting**: LLM-based query expansion for better retrieval

### Evaluation Requirements
1. **RAGAS Metrics**: Faithfulness, Context Precision, Answer Relevancy
2. **Baseline Comparison**: Measure improvements over naive retrieval
3. **A/B Testing**: Compare chunking strategies and retrieval methods
4. **Failure Documentation**: Track and analyze system failures

### Deployment Requirements
1. **Streamlit UI**: Interactive chat interface
2. **Docker Support**: Containerized deployment
3. **Streamlit Cloud**: Public deployment for demo
4. **GitHub Repository**: Portfolio-ready with documentation

## Functional Requirements

### FR-1: Document Ingestion
- Download SEC 10-K filings for 5 companies (AAPL, MSFT, GOOGL, AMZN, NVDA)
- Parse HTML/XML filing documents
- Clean and normalize text content
- Support incremental document updates

### FR-2: Document Processing
- Chunk documents using recursive character splitter
- Configurable chunk size (default: 1000 characters)
- Overlap between chunks (default: 200 characters)
- Preserve metadata (company, filing date, section)

### FR-3: Embedding Generation
- Generate embeddings using Ollama nomic-embed-text
- Store embeddings in Qdrant vector database
- Support metadata filtering at query time
- Persist embeddings to disk

### FR-4: Hybrid Retrieval
- Dense retrieval via vector similarity search
- Sparse retrieval via BM25 keyword matching
- Ensemble retrieval with Reciprocal Rank Fusion
- Configurable weights for each retriever

### FR-5: Query Rewriting
- LLM-based query expansion (generate 3-5 search queries)
- HyDE (Hypothetical Document Embedding) option
- Ambiguous query detection and clarification

### FR-6: Reranking
- CrossEncoder reranking with BAAI/bge-reranker-base
- Retrieve top-20, rerank to top-5
- Score normalization and filtering

### FR-7: Answer Generation
- LLM generation using Ollama qwen2.5:7b
- Strict source attribution and citation
- Streaming response support
- Refusal when context insufficient

### FR-8: Evaluation
- RAGAS framework integration
- Ollama as judge model for metrics (default; optionally a free hosted API via `JUDGE_MODEL`/`JUDGE_BASE_URL`/`JUDGE_API_KEY`)
- Automated evaluation pipeline
- Comparison report generation

### FR-9: User Interface
- Streamlit chat interface
- Real-time streaming responses
- Source document highlighting
- Evaluation metrics display in sidebar

### FR-10: Monitoring
- Langfuse integration for tracing
- Cost tracking per query
- Latency monitoring
- Error logging

## Non-Functional Requirements

### Performance
| Metric | Target | Measurement |
|--------|--------|-------------|
| Query Latency | <500ms p95 | End-to-end response time |
| Embedding Latency | <100ms | Per document embedding |
| Reranking Latency | <200ms | Per query reranking |
| Startup Time | <30s | Application cold start |

### Scalability
| Metric | Target | Notes |
|--------|--------|-------|
| Document Count | 100+ filings | Current: 15 |
| Vector Count | 50,000+ | Current: ~5,000 |
| Concurrent Users | 10+ | Streamlit limitation |
| Query Throughput | 100 queries/min | With caching |

### Reliability
| Metric | Target | Notes |
|--------|--------|-------|
| Uptime | 99% | Streamlit Cloud SLA |
| Error Rate | <1% | Failed queries |
| Recovery Time | <5min | From failure |

### Security
- No secrets in code (use .env files)
- API keys stored in environment variables
- No sensitive data in logs
- HTTPS for all external connections

## Constraints

### Technical Constraints
1. **Ollama Local**: All LLM inference runs locally (judge may optionally use a free hosted API via `JUDGE_API_KEY`)
2. **Docker Required**: Qdrant runs in Docker container
3. **Python 3.11+**: Required for latest LangChain features
4. **Memory**: 16GB RAM recommended for local models

### Budget Constraints
1. **No Cloud APIs**: All inference local via Ollama (judge may optionally use a free hosted OpenAI-compatible API — `JUDGE_API_KEY`; still no paid dependencies)
2. **Free Tier Only**: Streamlit Cloud free tier
3. **Open Source Only**: No paid dependencies

### Time Constraints
1. **4 Weeks**: Complete implementation timeline
2. **MVP First**: Core features before polish
3. **Documentation**: Required for portfolio

## Success Criteria

### Must Have (MVP)
- [ ] SEC 10-K documents ingested and chunked
- [ ] Qdrant vector store operational
- [ ] Basic retrieval working
- [ ] Hybrid retrieval (BM25 + Dense)
- [ ] Reranking implemented
- [ ] RAGAS evaluation running
- [ ] Streamlit UI functional
- [ ] Docker deployment working
- [ ] README with metrics

### Should Have
- [ ] Query rewriting
- [ ] Streaming responses
- [ ] Langfuse monitoring
- [ ] A/B testing notebook
- [ ] Failure analysis

### Nice to Have
- [ ] HyDE retrieval
- [ ] Multi-turn conversation
- [ ] Advanced caching
- [ ] Performance optimization

## Out of Scope

1. **Fine-tuning**: No model fine-tuning
2. **Training**: No embedding model training

*Multi-tenant, real-time ingestion, and authentication moved to Phase 2 (see below): multi-tenant via `tenant_id` payload filtering on a single collection, incremental ingestion, and API auth via static keys.*

## Assumptions

1. User has Docker installed and running
2. User has Ollama installed with required models
3. User has 16GB+ RAM for local inference
4. User has GitHub account for deployment
5. SEC EDGAR access is available (rate limits apply)

## Dependencies

### External Services
- SEC EDGAR API (for document download)
- Ollama (for local LLM inference)
- Docker Hub (for Qdrant image)
- Streamlit Cloud (for deployment)
- Langfuse (for monitoring)

### Python Packages
- langchain, langchain-community, langchain-ollama
- qdrant-client
- ragas
- streamlit
- sec-edgar-downloader
- rank-bm25
- sentence-transformers

## Phase 2 (in progress, 2026-09-16)

Planned enhancements on top of the MVP, currently in progress (designed, not yet verified).

1. **Multi-tenant support** — single Qdrant collection `sec_filings`, chunks tagged with `tenant_id` payload; retrieval filters by tenant. New `TENANT_ID` env (default `"default"`). No collection-per-tenant.
2. **Real-time document ingestion** — `scripts/ingest_one.py` ingests individual filings incrementally (parse → chunk → upsert, no collection recreation). `src/store.py` gains an idempotent incremental upsert (delete existing points for a source, keyed by hash of source + chunk_index). `scripts/ingest_index.py` gains a `--tenant` flag; full-rebuild path retained for batch rebuilds.
3. **Advanced caching (semantic cache)** — `src/cache.py` `SemanticCache` backed by a Qdrant `semantic_cache` collection using `EmbeddingGenerator`. `lookup(query)` returns a cached response when top-1 cosine similarity ≥ `CACHE_THRESHOLD` (default 0.92). Served before retrieval, written after generation in `RAGPipeline.query()`; bypassed for the `baseline` strategy. Wiped on re-ingest.
4. **API gateway** — FastAPI `app/api.py`: `POST /query`, `POST /ingest`, `GET /health`. Auth = `X-API-Key` header against comma-separated `API_KEYS` env (401 otherwise). Pipeline cached per strategy at module scope. New `api` service in docker-compose (port 8000).

New env vars: `TENANT_ID`, `CACHE_ENABLED`, `CACHE_THRESHOLD`, `API_KEYS` (comma-separated). `.env` and `data/` are git-ignored — never commit them.

## Glossary

| Term | Definition |
|------|-----------|
| RAG | Retrieval-Augmented Generation |
| RRF | Reciprocal Rank Fusion |
| BM25 | Best Matching 25 (keyword retrieval algorithm) |
| CrossEncoder | Neural reranking model |
| HyDE | Hypothetical Document Embedding |
| RAGAS | Retrieval Augmented Generation Assessment |
| 10-K | Annual SEC filing for public companies |
