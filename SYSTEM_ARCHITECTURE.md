# System Architecture

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Interface"
        UI[Streamlit Chat UI]
    end
    
    subgraph "Application Layer"
        APP[RAG Application]
        RW[Query Rewriter]
        RET[Hybrid Retriever]
        RR[Reranker]
        GEN[LLM Generator]
        MON[Langfuse Monitor]
    end
    
    subgraph "Retrieval Layer"
        BM25[BM25 Retriever]
        DENSE[Dense Retriever]
        ENS[Ensemble Retriever]
    end
    
    subgraph "Storage Layer"
        QDRANT[(Qdrant Vector DB)]
        EMBED[Embeddings Store]
    end
    
    subgraph "External Services"
        OLLAMA[Ollama Server]
        SEC[SEC EDGAR API]
        LANGFUSE[Langfuse Cloud]
    end
    
    UI --> APP
    APP --> RW
    RW --> RET
    RET --> ENS
    ENS --> BM25
    ENS --> DENSE
    BM25 --> QDRANT
    DENSE --> QDRANT
    RET --> RR
    RR --> GEN
    GEN --> OLLAMA
    APP --> MON
    MON --> LANGFUSE
    QDRANT --> EMBED
```

## Component Diagram

```mermaid
classDiagram
    class RAGApplication {
        +query(user_query: str) Response
        +ingest_documents()
        +evaluate()
    }
    
    class QueryRewriter {
        +rewrite(query: str) List~str~
        +expand(query: str) List~str~
    }
    
    class HybridRetriever {
        +retrieve(query: str) List~Document~
        +set_weights(bm25: float, dense: float)
    }
    
    class Reranker {
        +rerank(query: str, docs: List~Document~) List~Document~
        +score(query: str, doc: Document) float
    }
    
    class LLMGenerator {
        +generate(query: str, context: str) str
        +stream(query: str, context: str) Generator
    }
    
    class VectorStore {
        +search(query_embedding: List~float~) List~Document~
        +upsert(documents: List~Document~)
        +filter(metadata: Dict)
    }
    
    class Evaluator {
        +run_evaluation() Dict
        +compare_baselines() DataFrame
        +analyze_failures() List
    }
    
    RAGApplication --> QueryRewriter
    RAGApplication --> HybridRetriever
    RAGApplication --> Reranker
    RAGApplication --> LLMGenerator
    RAGApplication --> Evaluator
    HybridRetriever --> VectorStore
```

## Data Flow

### Ingestion Pipeline

```mermaid
flowchart LR
    A[SEC EDGAR] -->|Download| B[Raw Filings]
    B -->|Parse HTML/XML| C[Clean Text]
    C -->|Chunk| D[Document Chunks]
    D -->|Embed| E[Embeddings]
    E -->|Store| F[Qdrant DB]
    D -->|Metadata| F
```

### Query Pipeline

```mermaid
flowchart LR
    A[User Query] -->|Rewrite| B[Expanded Queries]
    B -->|Hybrid Search| C[Top-20 Candidates]
    C -->|Rerank| D[Top-5 Results]
    D -->|Format Context| E[Prompt]
    E -->|Generate| F[LLM Response]
    F -->|Cite Sources| G[Final Answer]
```

## Infrastructure

### Docker Compose

```mermaid
graph TB
    subgraph "Docker Network"
        subgraph "Qdrant Container"
            QDRANT_API[REST API :6333]
            QDRANT_GRPC[gRPC :6334]
            QDRANT_DATA[Persistent Storage]
        end
        
        subgraph "Ollama Container"
            OLLAMA_API[REST API :11434]
            OLLAMA_MODELS[Model Storage]
        end
    end
    
    subgraph "Host Machine"
        PYTHON[Python Application]
        STREAMLIT[Streamlit UI]
    end
    
    PYTHON --> QDRANT_API
    PYTHON --> OLLAMA_API
    STREAMLIT --> PYTHON
```

### Network Configuration

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| Qdrant REST | 6333 | HTTP | Vector DB API |
| Qdrant gRPC | 6334 | gRPC | Bulk operations |
| Ollama | 11434 | HTTP | LLM inference |
| Streamlit | 8501 | HTTP | Web UI |

## Technology Stack

### Core Components

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| LLM | Ollama + qwen2.5:7b | 0.5.7+ | Answer generation |
| Embeddings | Ollama + nomic-embed-text | 0.5.7+ | Vector creation |
| Eval Judge | Ollama qwen2.5:7b (default) or free OpenAI-compatible API (`JUDGE_*`) | 0.5.7+ | RAGAS metric judging |
| Vector DB | Qdrant | 1.13.0 | Vector storage |
| Framework | LangChain | 0.3.18 | Orchestration |
| Evaluation | RAGAS | Latest | Quality metrics |
| UI | Streamlit | Latest | User interface |
| Monitoring | Langfuse | Latest | Tracing |

### Python Dependencies

```
langchain==0.3.18
langchain-community==0.3.18
langchain-ollama==0.2.3
langchain-core==0.3.34
qdrant-client==1.13.0
ragas
streamlit
sec-edgar-downloader
rank-bm25
sentence-transformers
langfuse
python-dotenv
```

## Security Architecture

### Secret Management

```mermaid
graph LR
    A[.env File] -->|Load| B[Python Application]
    B -->|Use| C[API Calls]
    D[Environment Variables] -->|Override| B
    E[Streamlit Secrets] -->|Production| B
```

### Security Rules

1. **Never commit .env files** - Use .env.example as template
2. **API keys in environment** - Not in code
3. **No secrets in logs** - Sanitize all output
4. **HTTPS only** - For external connections
5. **Local inference** - No data leaves machine

## Performance Architecture

### Latency Budget

```
Total Query Latency: <500ms (target)

Query Rewriting:     ~100ms  (20%)
BM25 Retrieval:      ~50ms   (10%)
Dense Retrieval:     ~100ms  (20%)
Ensemble Fusion:     ~20ms   (4%)
Reranking:           ~150ms  (30%)
LLM Generation:      ~80ms   (16%)
```

### Caching Strategy

```mermaid
graph LR
    A[Query] -->|Embed| SC[Semantic Cache]
    SC -->|cosine sim >= threshold| C[Cached Response]
    SC -->|miss| D[Full Pipeline]
    D -->|Store| E[Cache]
    E -->|Return| C
```

### Memory Management

| Component | Memory | Notes |
|-----------|--------|-------|
| Qdrant | ~500MB | Vector storage |
| Ollama Models | ~5GB | qwen2.5:7b + nomic |
| Python App | ~500MB | Application code |
| **Total** | **~6GB** | Recommended: 16GB |

## Scalability Considerations

### Current Scale
- 15 documents (5 companies × 3 years)
- ~5,000 chunks
- ~5,000 vectors
- Single user

### Growth Path
| Metric | Current | 10x | 100x |
|--------|---------|-----|------|
| Documents | 15 | 150 | 1,500 |
| Vectors | 5K | 50K | 500K |
| Storage | 100MB | 1GB | 10GB |
| Memory | 6GB | 8GB | 16GB |

### Scaling Strategies
1. **Horizontal**: Multiple Qdrant replicas
2. **Vertical**: Larger Qdrant instance
3. **Caching**: Redis for frequent queries
4. **Batching**: Async document processing

## Monitoring Architecture

```mermaid
graph TB
    subgraph "Application"
        A[RAG Pipeline] -->|Trace| B[Langfuse SDK]
    end
    
    subgraph "Langfuse Cloud"
        B -->|Send| C[Trace Storage]
        C --> D[Dashboard]
        C --> E[Analytics]
        C --> F[Alerts]
    end
    
    subgraph "Metrics"
        G[Latency] --> D
        H[Cost] --> D
        I[Errors] --> D
        J[Quality] --> D
    end
```

### Traced Events

| Event | Data Captured |
|-------|---------------|
| Query | Input, timestamp, user |
| Retrieval | Chunks, scores, latency |
| Generation | Prompt, response, tokens |
| Error | Exception, stack trace |

## Deployment Architecture

### Local Development

```mermaid
graph TB
    A[Developer Machine] --> B[Docker Desktop]
    B --> C[Qdrant Container]
    B --> D[Ollama Container]
    A --> E[Python Virtual Env]
    E --> F[Streamlit Server]
```

### Production (Streamlit Cloud)

```mermaid
graph TB
    A[GitHub Repo] -->|Push| B[Streamlit Cloud]
    B --> C[Build Container]
    C --> D[Deploy App]
    D --> E[Public URL]
    E --> F[User Browser]
    D --> G[Connected Services]
    G --> H[Ollama Local]
    G --> I[Qdrant Local]
```

## Failure Modes

### Failure Detection

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Ollama Down | Connection error | Restart container |
| Qdrant Down | Connection error | Restart container |
| Embedding Mismatch | Dimension error | Rebuild index |
| LLM Timeout | Timeout error | Retry with fallback |
| Memory OOM | Kill signal | Restart with smaller model |

### Fallback Strategy

```mermaid
graph TD
    A[Query] --> B{Primary Pipeline}
    B -->|Success| C[Return Answer]
    B -->|Failure| D{Fallback 1}
    D -->|Dense Only| E[Dense Retrieval]
    D -->|Failure| F{Fallback 2}
    F -->|BM25 Only| G[BM25 Retrieval]
    F -->|Failure| H[Error Response]
```

## Future Architecture

### Phase 2 Enhancements *(in progress as of 2026-09-16)*
1. **Multi-tenant (payload-filter)**: Single `sec_filings` collection, chunks tagged with `tenant_id` payload; retrieval filters by tenant via qdrant filter. No collection-per-tenant.
2. **Real-time Ingestion**: `scripts/ingest_one.py` parses → chunks → upserts individual filings without recreating the collection; `src/store.py` incremental upsert is idempotent (keyed by hash of source + chunk_index). `ingest_index.py` gains a `--tenant` flag.
3. **Advanced Caching (semantic cache)**: `src/cache.py` `SemanticCache` in a Qdrant `semantic_cache` collection, served before retrieval when top-1 cosine similarity ≥ `CACHE_THRESHOLD` (0.92); bypassed for the `baseline` strategy.
4. **API Gateway**: FastAPI `app/api.py` (`POST /query`, `POST /ingest`, `GET /health`), auth via `X-API-Key` against comma-separated `API_KEYS` env. Package + `api` service in docker-compose (port 8000).

### Phase 3 Enhancements *(implemented 2026-09-16)*
1. **Self-RAG (adaptive retrieval)**: `src/selfrag.py` — reranker cross-encoder score attached to docs (`rerank_score`). Best score below `SELF_RAG_MIN_CONFIDENCE` (0.3) → re-retrieve with wider `SELF_RAG_EXPAND_K` (40) + re-rerank; still below `SELF_RAG_REFUSE_BELOW` (0.15) → refuse to answer. Active for `full`/`rerank` strategies in `RAGPipeline.query()`.
2. **Guardrails**: `src/guardrails.py` — input guard regex-blocking prompt injections, PII, off-topic keywords (canned refusal); output guard refusing leaked PII / botched refusals. Wired into `RAGPipeline.query()`.
3. **Analytics Dashboard**: `app/streamlit_app.py` sidebar `Chat | Analytics` toggle — Qdrant chunk count, semantic-cache entries, session cache hit rate (tracked in `src/cache.py`), session latency, A/B results (`ab_report.md` → `summary.json` → run hint).
4. **A/B Report**: `scripts/ab_report.py` — reads `results_ab/scores_{strategy}_{metric}.json`, computes per-strategy/per-metric means, picks best strategy per metric, writes `ab_report.md`; `--no-write` prints only.
