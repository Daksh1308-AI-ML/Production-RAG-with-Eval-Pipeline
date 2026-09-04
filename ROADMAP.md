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
- [ ] Create project structure
- [ ] Set up Python virtual environment
- [ ] Install dependencies (requirements.txt)
- [ ] Configure Docker Compose (Qdrant + Ollama)
- [ ] Pull Ollama models (qwen2.5:7b, nomic-embed-text)
- [ ] Create .env file from .env.example
- [ ] Initialize git repository
- [ ] Create config.py with all configurations

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
- [ ] Implement SECIngestor class
- [ ] Download 10-K filings for AAPL, MSFT, GOOGL, AMZN, NVDA
- [ ] Parse HTML/XML to clean text
- [ ] Extract metadata (company, date, section)
- [ ] Store processed documents in data/processed/

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
- [ ] Implement DocumentChunker class
- [ ] Implement EmbeddingGenerator class
- [ ] Implement VectorStore class
- [ ] Create Qdrant collection
- [ ] Chunk and embed all documents
- [ ] Store in Qdrant
- [ ] Implement basic dense retrieval
- [ ] Test with sample queries

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
- [ ] Project structure created
- [ ] Docker containers running (Qdrant + Ollama)
- [ ] SEC filings downloaded and parsed
- [ ] Documents chunked (1000 chars, 200 overlap)
- [ ] Embeddings generated and stored in Qdrant
- [ ] Basic dense retrieval working
- [ ] Basic RAG chain functional

### Week 1 Metrics
| Metric | Target | Actual |
|--------|--------|--------|
| Documents downloaded | 15 | |
| Chunks created | ~5,000 | |
| Embedding latency | <100ms | |
| Retrieval latency | <200ms | |

---

## Week 2: Advanced Retrieval

### Objectives
- Implement hybrid retrieval (BM25 + Dense)
- Add query rewriting
- Implement reranking

### Day 8-10: Hybrid Retrieval

**Tasks**:
- [ ] Implement BM25Retriever integration
- [ ] Implement HybridRetriever class
- [ ] Configure EnsembleRetriever with RRF
- [ ] Set initial weights (BM25: 0.4, Dense: 0.6)
- [ ] Test hybrid retrieval
- [ ] Add metadata filtering

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
- [ ] Implement QueryRewriter class
- [ ] Add query expansion (3-5 queries)
- [ ] Implement should_rewrite logic
- [ ] Test with ambiguous queries
- [ ] Integrate with retrieval pipeline

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
- [ ] Implement Reranker class
- [ ] Load BAAI/bge-reranker-base model
- [ ] Implement rerank method
- [ ] Create compression retriever
- [ ] Test reranking pipeline
- [ ] Measure latency impact

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
- [ ] HybridRetriever with BM25 + Dense
- [ ] QueryRewriter with expansion
- [ ] Reranker with CrossEncoder
- [ ] Full retrieval pipeline working
- [ ] Metadata filtering functional

### Week 2 Metrics
| Metric | Target | Actual |
|--------|--------|--------|
| Hybrid retrieval latency | <300ms | |
| Query rewriting latency | <100ms | |
| Reranking latency | <200ms | |
| Total retrieval latency | <500ms | |

---

## Week 3: Evaluation Pipeline

### Objectives
- Set up RAGAS evaluation
- Create evaluation dataset
- Run baseline vs hybrid comparison
- Document failures

### Day 15-16: RAGAS Setup

**Tasks**:
- [ ] Implement RAGEvaluator class
- [ ] Configure Ollama as judge model
- [ ] Create evaluation dataset template
- [ ] Write 30 initial QA pairs
- [ ] Test RAGAS integration

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
- [ ] Create 100+ QA pairs
- [ ] Include ground truth answers
- [ ] Categorize by question type
- [ ] Add metadata (company, difficulty)
- [ ] Review and validate questions
- [ ] Save to data/evaluation/

**Deliverables**:
- 100+ QA pairs with ground truth
- Evaluation dataset JSON
- Question type distribution

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
- [ ] 100+ QA evaluation dataset
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
- [ ] Environment working
- [ ] Data ingested
- [ ] Basic RAG functional

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
