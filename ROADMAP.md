# Implementation Roadmap

## Overview

4-week implementation plan for Production RAG with Eval Pipeline. Each week has specific deliverables and acceptance criteria.

## Week 1: Data Pipeline & Basic RAG

### Objectives
- Set up development environment
- Download and process SEC 10-K filings
- Implement basic RAG pipeline

### Day 1-2: Project Setup

**Tasks**:
- [x] Create project structure
- [x] Set up Python virtual environment
- [x] Install dependencies (requirements.txt)
- [x] Configure Docker Compose (Qdrant + Ollama)
- [x] Pull Ollama models (qwen2.5:7b, nomic-embed-text)
- [x] Create .env file
- [x] Initialize git repository
- [x] Create config.py with all configurations

**Deliverables**:
- Working dev environment
- Docker containers running
- Configuration module ready

**Verification**:
```bash
# Verify Ollama
curl http://localhost:11434/api/tags

# Verify Qdrant
curl http://localhost:6333/dashboard/

# Verify Python environment
python -c "from src.config import config; print(config)"
```

### Day 3-4: SEC Data Ingestion

**Tasks**:
- [x] Implement SECIngestor class
- [x] Download 10-K filings for AAPL, MSFT, GOOGL, AMZN, NVDA
- [x] Parse HTML/XML to clean text
- [x] Extract metadata (company, date, section)
- [x] Store processed documents in data/processed/

**Deliverables**:
- 15 clean 10-K filings (5 companies × 3 years)
- Parsed text with metadata
- Ingestion script

**Verification**:
```bash
# Check downloaded files
ls data/sec_filings/

# Check processed files
python -c "from src.ingest import SECIngestor; i = SECIngestor(); print(i.download_all())"
```

### Day 5-7: Chunking, Embedding, Basic Retrieval

**Tasks**:
- [x] Implement DocumentChunker class
- [x] Implement EmbeddingGenerator class
- [x] Implement VectorStore class
- [x] Create Qdrant collection
- [x] Chunk and embed all documents
- [x] Store in Qdrant
- [x] Implement basic dense retrieval
- [x] Test with sample queries

**Deliverables**:
- Chunked documents with embeddings
- Qdrant vector store populated
- Basic retrieval working

**Verification**:
```bash
# Test embedding
python -c "from src.embed import EmbeddingGenerator; e = EmbeddingGenerator(); print(e.embed_query('test')[:5])"

# Test retrieval
python -c "
from src.store import VectorStore
vs = VectorStore()
results = vs.search('What are Apple revenue sources?')
print(f'Retrieved {len(results)} documents')
"
```

### Week 1 Deliverables Checklist
- [x] Project structure created
- [x] Docker containers running (Qdrant + Ollama)
- [x] SEC filings downloaded and parsed
- [x] Documents chunked (1000 chars, 200 overlap)
- [x] Embeddings generated and stored in Qdrant
- [x] Basic dense retrieval working
- [x] Basic RAG chain functional

### Week 1 Metrics
| Metric | Target | Actual |
|--------|--------|--------|
| Documents downloaded | 15 | 15 |
| Chunks created | ~5,000 | 39,565 |
| Embedding latency | <100ms | ~52 chunks/s bulk (~19ms/chunk) |
| Retrieval latency | <200ms | 68–102ms per query (first query 2.2s, cold embed) |

---

## Week 2: Advanced Retrieval

### Objectives
- Implement hybrid retrieval (BM25 + Dense)
- Add query rewriting
- Implement reranking

### Day 8-10: Hybrid Retrieval

**Tasks**:
- [x] Implement BM25Retriever integration
- [x] Implement HybridRetriever class
- [x] Configure EnsembleRetriever with RRF
- [x] Set initial weights (BM25: 0.4, Dense: 0.6)
- [x] Test hybrid retrieval
- [x] Add metadata filtering

**Deliverables**:
- HybridRetriever class working
- BM25 + Dense fusion via RRF
- Metadata filtering functional

**Verification**:
```bash
# Test hybrid retrieval
python -c "
from src.retrieve import HybridRetriever
r = HybridRetriever()
results = r.retrieve('What are Apple risk factors?')
print(f'Hybrid retrieved {len(results)} documents')
"
```

### Day 11-12: Query Rewriting

**Tasks**:
- [x] Implement QueryRewriter class
- [x] Add query expansion (3-5 queries)
- [x] Implement should_rewrite logic
- [x] Test with ambiguous queries
- [x] Integrate with retrieval pipeline

**Deliverables**:
- QueryRewriter class working
- Query expansion generating multiple queries
- Integration with hybrid retrieval

**Verification**:
```bash
# Test query rewriting
python -c "
from src.query_rewrite import QueryRewriter
qr = QueryRewriter()
queries = qr.rewrite('Tell me about risks')
print(f'Rewritten to {len(queries)} queries')
"
```

### Day 13-14: Reranking

**Tasks**:
- [x] Implement Reranker class
- [x] Load BAAI/bge-reranker-base model
- [x] Implement rerank method
- [x] Test reranking pipeline
- [x] Create compression retriever
  - Compression = the `Reranker` itself (top-20 → top-5); a separate LangChain `ContextualCompressionRetriever` wrapper is redundant (YAGNI)
- [x] Measure latency impact
  - Verified via `scripts/validate_hybrid.py` (PASS): rerank 0.34–0.58s on CPU with `ms-marco-MiniLM-L6-v2`, 2.0–4.8s with `bge-reranker-base`; `bge-reranker-base` ~12–15s to load after warm-up
  - Pre-download only needed model files (`snapshot_download(allow_patterns=...)`) — full `snapshot_download` of `ms-marco-MiniLM-L6-v2` pulls ~865 MB of redundant ONNX/OpenVINO/Flax/PyTorch copies and stalls

**Deliverables**:
- Reranker class working
- Top-20 → Top-5 reranking
- Latency <200ms added

**Verification**:
```bash
# Test reranking
python -c "
from src.rerank import Reranker
from langchain.schema import Document
rr = Reranker()
docs = [Document(page_content='test', metadata={}) for _ in range(10)]
reranked = rr.rerank('test query', docs, top_n=5)
print(f'Reranked to {len(reranked)} documents')
"
```

### Week 2 Deliverables Checklist
- [x] HybridRetriever with BM25 + Dense
- [x] QueryRewriter with expansion
- [x] Reranker with CrossEncoder
- [x] Full retrieval pipeline working
- [x] Metadata filtering functional

### Week 2 Metrics
| Metric | Target | Actual |
|--------|--------|--------|
| Hybrid retrieval latency | <300ms | 115–183ms |
| Query rewriting latency | <100ms | 13.7s (LLM-generated via Ollama) |
| Reranking latency | <200ms | 0.34–0.58s (MiniLM), 2.0–4.8s (bge) |
| Total retrieval latency | <500ms | 115–183ms pre-rerank (rewrite: +13.7s llm, fusion 3.7s) |

> Retargeting note: `<100ms`/`<200ms` assume a fast hosted cross-encoder; local CPU inference is slower. Rerank with the smaller cached `ms-marco-MiniLM-L6-v2` meets 0.3–0.6s.

---

## Week 3: Evaluation Pipeline

### Objectives
- Set up RAGAS evaluation
- Create evaluation dataset
- Run baseline vs hybrid comparison
- Document failures

### Day 15-16: RAGAS Setup

**Tasks**:
- [x] Implement RAGEvaluator class
- [x] Configure judge model (Ollama `qwen2.5:7b` by default; free OpenAI-compatible API via `JUDGE_*` vars)
- [x] Create evaluation dataset template (`scripts/build_eval_dataset.py` → `data/evaluation/eval_dataset.json`)
- [x] Write 30 initial QA pairs (5 tickers × 6; factual + analytical, easy/medium/hard)
- [x] Test RAGAS integration (smoke eval: `python -m src.eval --limit 2 --strategies baseline --metrics faithfulness` → baseline faithfulness **0.67** via pinned OpenRouter judge `nvidia/nemotron-3-super-120b-a12b:free`)

**Deliverables**:
- RAGEvaluator class working
- Evaluation dataset started
- RAGAS metrics running

**Verification**:
```bash
# Test RAGAS
python -c "
from src.eval import RAGEvaluator
ev = RAGEvaluator()
print('RAGAS evaluator initialized')
"
```

### Day 17-19: Evaluation Dataset Creation

**Tasks**:
- [x] Create 100+ QA pairs
- [x] Include ground truth answers
- [x] Categorize by question type
- [x] Add metadata (company, difficulty)
- [x] Review and validate questions
- [x] Save to data/evaluation/

**Notes**:
- 103 QA pairs built: 20 each for AAPL/MSFT/AMZN/NVDA, 19 for GOOGL, plus 4 cross-company comparative questions.
- Distribution: factual 67 / analytical 21 / comparative 12 / summary 3; easy 25 / medium 51 / hard 27.
- Every `reference_contexts` entry is verified verbatim (whitespace-collapsed substring) against `data/processed/filings.json` by `scripts/build_eval_dataset.py`; the builder asserts this on every run.

**Deliverables**:
- [x] 100+ QA pairs with ground truth
- [x] Evaluation dataset JSON
- [x] Question type distribution

**Verification**:
```bash
# Check dataset
python -c "
import json
with open('data/evaluation/eval_dataset.json') as f:
    data = json.load(f)
print(f'Dataset has {len(data)} QA pairs')
"
```

### Day 20-21: A/B Testing & Failure Analysis

**Tasks**:
- [ ] Run baseline evaluation (dense-only)
- [ ] Run hybrid evaluation
- [ ] Run hybrid + reranking evaluation
- [ ] Run full pipeline evaluation
- [ ] Compare results
- [ ] Document failures
- [ ] Create analysis notebook

**Deliverables**:
- Evaluation results comparison
- Failure analysis report
- Improvement recommendations

**Verification**:
```bash
# Run evaluation
python -m src.eval --dataset data/evaluation/eval_dataset.json --output results/
```

### Week 3 Deliverables Checklist
- [ ] RAGAS evaluation pipeline working
- [x] 100+ QA evaluation dataset
- [ ] Baseline vs hybrid comparison
- [ ] Failure analysis documented
- [ ] A/B test results

### Week 3 Metrics
| Metric | Baseline | Target | Actual |
|--------|----------|--------|--------|
| Faithfulness | 55% | 85%+ | |
| Answer Relevancy | 65% | 80%+ | |
| Context Precision | 60% | 80%+ | |
| Context Recall | 65% | 80%+ | |

---

## Week 4: Deployment & Polish

### Objectives
- Build Streamlit UI
- Add Langfuse monitoring
- Deploy to Streamlit Cloud
- Write comprehensive README

### Day 22-24: Streamlit UI

**Tasks**:
- [ ] Create Streamlit app structure
- [ ] Implement chat interface
- [ ] Add streaming responses
- [ ] Display source citations
- [ ] Add evaluation metrics sidebar
- [ ] Style with custom CSS

**Deliverables**:
- Working Streamlit chat UI
- Streaming responses
- Source highlighting
- Metrics display

**Verification**:
```bash
# Run Streamlit app
streamlit run app/streamlit_app.py
```

### Day 25-26: Monitoring & Docker

**Tasks**:
- [ ] Integrate Langfuse tracing
- [ ] Add latency monitoring
- [ ] Add error logging
- [ ] Create Dockerfile
- [ ] Update docker-compose.yml
- [ ] Test Docker deployment

**Deliverables**:
- Langfuse integration working
- Docker deployment functional
- Monitoring dashboard

**Verification**:
```bash
# Build and run Docker
docker-compose up -d
docker-compose ps
```

### Day 27-28: Documentation & Deployment

**Tasks**:
- [ ] Write README.md
- [ ] Add architecture diagram
- [ ] Document setup instructions
- [ ] Include metrics and results
- [ ] Create demo GIF
- [ ] Deploy to Streamlit Cloud
- [ ] Final testing

**Deliverables**:
- Comprehensive README
- Deployed application
- Portfolio-ready project

**Verification**:
```bash
# Deploy to Streamlit Cloud
git push origin main
# Verify deployment at https://share.streamlit.io/
```

### Week 4 Deliverables Checklist
- [ ] Streamlit UI functional
- [ ] Langfuse monitoring active
- [ ] Docker deployment working
- [ ] README comprehensive
- [ ] Deployed to Streamlit Cloud
- [ ] Demo GIF recorded

### Week 4 Metrics
| Metric | Target | Actual |
|--------|--------|--------|
| UI latency | <600ms | |
| Docker build | <5min | |
| Deployment success | 100% | |
| Documentation completeness | 100% | |

---

## Final Project Checklist

### Must Have (MVP)
- [ ] SEC 10-K documents ingested (15 filings)
- [ ] Qdrant vector store operational
- [ ] Hybrid retrieval (BM25 + Dense)
- [ ] Reranking implemented
- [ ] Query rewriting working
- [ ] RAGAS evaluation complete
- [ ] Streamlit UI functional
- [ ] Docker deployment working
- [ ] README with metrics

### Should Have
- [ ] Langfuse monitoring
- [ ] Streaming responses
- [ ] A/B testing notebook
- [ ] Failure analysis
- [ ] Demo GIF

### Nice to Have
- [ ] HyDE retrieval
- [ ] Multi-turn conversation
- [ ] Advanced caching
- [ ] Performance optimization

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Ollama model download fails | High | Low | Pre-download models |
| Hugging Face reranker download stalls | High | Medium | Pre-download only needed files via `snapshot_download(allow_patterns=...)` (README §5); verified for `bge-reranker-base` + `ms-marco-MiniLM-L6-v2` |
| Qdrant connection issues | High | Medium | Check Docker status |
| RAGAS evaluation slow | Medium | Medium | Use smaller eval set |
| Streamlit Cloud deployment fails | Medium | Low | Test locally first |
| Memory issues with models | High | Low | Use smaller models |
| SEC rate limiting | Medium | Medium | Add delays between requests |

---

## Dependencies

### External
- SEC EDGAR API (for data)
- Docker Hub (for images)
- Streamlit Cloud (for deployment)
- Langfuse Cloud (for monitoring)

### Internal
- Ollama running locally
- Qdrant Docker container
- Python 3.11+ environment
- GitHub account

---

## Success Criteria

### Week 1 Success
- [x] Environment working
- [x] Data ingested
- [x] Basic RAG functional

### Week 2 Success
- [ ] Hybrid retrieval working
- [ ] Query rewriting functional
- [ ] Reranking implemented

### Week 3 Success
- [ ] RAGAS evaluation complete
- [ ] Metrics improved over baseline
- [ ] Failures documented

### Week 4 Success
- [ ] UI deployed
- [ ] Monitoring active
- [ ] README complete
- [ ] Portfolio-ready

---

## Next Phase (Post-MVP)

### Phase 2 Enhancements
1. Multi-tenant support
2. Real-time document ingestion
3. Advanced caching (semantic cache)
4. API gateway for external access

### Phase 3 Enhancements
1. Self-RAG (adaptive retrieval)
2. Guardrails for sensitive content
3. Analytics dashboard
4. A/B testing framework
