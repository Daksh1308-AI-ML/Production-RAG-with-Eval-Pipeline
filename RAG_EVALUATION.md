# RAG Evaluation Framework

## Overview

This document defines the evaluation methodology for the Production RAG system using RAGAS (Retrieval Augmented Generation Assessment). The evaluation measures retrieval quality, generation faithfulness, and overall system performance.

## Evaluation Metrics

### Primary Metrics

| Metric | Definition | Target | Measurement |
|--------|-----------|--------|-------------|
| **Faithfulness** | Is the answer grounded in retrieved context? | 90%+ | Claims supported / total claims |
| **Answer Relevancy** | Does the answer address the question? | 88%+ | Semantic similarity of generated questions |
| **Context Precision** | Are retrieved chunks relevant? | 85%+ | Useful chunks / total chunks |
| **Context Recall** | Did we retrieve all needed information? | 85%+ | Ground truth claims covered |

### Secondary Metrics

| Metric | Definition | Target | Measurement |
|--------|-----------|--------|-------------|
| **Latency** | End-to-end response time | <500ms p95 | Time measurement |
| **Hallucination Rate** | Answers not in context | <5% | Manual evaluation |
| **Citation Accuracy** | Sources correctly referenced | 90%+ | Manual verification |

## RAGAS Metrics Deep Dive

### 1. Faithfulness

**What it measures**: Whether the generated answer is supported by the retrieved context.

**How it works**:
1. Extract atomic claims from the answer
2. For each claim, check if it's supported by any context chunk
3. Score = supported claims / total claims

**Interpretation**:
- 0.9+ = Excellent (most claims grounded)
- 0.7-0.9 = Good (some minor hallucinations)
- <0.7 = Poor (significant hallucination issues)

**Common causes of low faithfulness**:
- Weak prompt instructions ("answer freely")
- LLM overriding retrieved context
- Context too vague to support specific claims

**Improvement strategies**:
- Strengthen system prompt ("Answer ONLY from context")
- Add "I don't know" fallback
- Improve retrieval to provide more specific context

### 2. Answer Relevancy

**What it measures**: Whether the answer actually addresses the user's question.

**How it works**:
1. Generate N potential questions from the answer
2. Calculate cosine similarity between original question and generated questions
3. High similarity = relevant answer

**Interpretation**:
- 0.9+ = Excellent (answer directly addresses question)
- 0.7-0.9 = Good (answer partially relevant)
- <0.7 = Poor (answer off-topic)

**Common causes of low relevancy**:
- Answer includes unnecessary information
- Answer is incomplete
- Answer addresses a different question

**Improvement strategies**:
- Tighten prompt to focus on question
- Add answer length constraints
- Improve retrieval for question-specific context

### 3. Context Precision

**What it measures**: Signal-to-noise ratio of retrieved chunks.

**How it works**:
1. For each retrieved chunk, determine if it's useful for answering the question
2. Weight by rank (higher-ranked useful chunks score more)
3. Score = weighted useful chunks / total chunks

**Interpretation**:
- 0.9+ = Excellent (most chunks relevant)
- 0.7-0.9 = Good (some irrelevant chunks)
- <0.7 = Poor (too much noise)

**Common causes of low precision**:
- Embedding model mismatch
- Retrieval returning tangentially related chunks
- No reranking to filter noise

**Improvement strategies**:
- Add reranking step
- Improve embedding model
- Tune retrieval k (fewer but better chunks)

### 4. Context Recall

**What it measures**: Whether all necessary information was retrieved.

**How it works**:
1. Extract claims from ground truth answer
2. Check if each claim is supported by retrieved context
3. Score = supported ground truth claims / total claims

**Interpretation**:
- 0.9+ = Excellent (all info retrieved)
- 0.7-0.9 = Good (most info retrieved)
- <0.7 = Poor (missing critical information)

**Common causes of low recall**:
- Retrieval k too small
- Chunking splitting important information
- Query too vague for specific retrieval

**Improvement strategies**:
- Increase retrieval k
- Improve chunking (larger chunks or better overlap)
- Add query rewriting/expansion

## Evaluation Dataset Design

### Dataset Structure

```json
{
  "question": "What are the main risk factors for Apple?",
  "contexts": ["chunk1 text", "chunk2 text"],
  "answer": "Apple faces risks from...",
  "ground_truth": "Apple's main risk factors include..."
}
```

### Dataset Creation Guidelines

#### Question Types (100+ total)

| Type | Count | Example |
|------|-------|---------|
| Factual | 30 | "What was Apple's revenue in 2024?" |
| Analytical | 25 | "How does Microsoft's cloud growth compare to AWS?" |
| Comparative | 20 | "What are the differences in risk factors between Apple and Google?" |
| Summary | 15 | "Summarize Amazon's business strategy." |
| Edge Cases | 10 | "What is the filing date of Apple's 2024 10-K?" |

#### Ground Truth Creation

1. **Manual annotation**: Human-written reference answers
2. **Source verification**: Every claim traced to specific document section
3. **Consensus review**: Multiple annotators verify accuracy
4. **Version control**: Track changes to ground truth

### Sample Evaluation Dataset

```json
[
  {
    "question": "What are Apple's main sources of revenue?",
    "contexts": [],
    "answer": "",
    "ground_truth": "Apple's main revenue sources are iPhone (52% of revenue), Services (22%), Mac (8%), iPad (7%), and Wearables (10%).",
    "metadata": {
      "company": "AAPL",
      "filing_year": 2024,
      "section": "financial_statements",
      "difficulty": "easy"
    }
  },
  {
    "question": "Compare the debt levels of Apple and Microsoft.",
    "contexts": [],
    "answer": "",
    "ground_truth": "Apple had $111 billion in total debt while Microsoft had $41 billion as of their respective 2024 fiscal years.",
    "metadata": {
      "company": "multiple",
      "filing_year": 2024,
      "section": "financial_statements",
      "difficulty": "medium"
    }
  },
  {
    "question": "What regulatory risks does Google face?",
    "contexts": [],
    "answer": "",
    "ground_truth": "Google faces antitrust litigation from the DOJ, potential remedies including structural changes, and ongoing regulatory scrutiny in multiple jurisdictions.",
    "metadata": {
      "company": "GOOGL",
      "filing_year": 2024,
      "section": "risk_factors",
      "difficulty": "medium"
    }
  }
]
```

## Evaluation Pipeline

### Step 1: Baseline Evaluation

Run evaluation on naive retrieval (dense-only, no reranking):

```python
# Baseline: Dense retrieval, top-5, no reranking
baseline_results = evaluator.run_evaluation(
    retrieval_method="dense",
    k=5,
    reranking=False
)
```

**Expected baseline metrics**:
- Faithfulness: 55%
- Answer Relevancy: 65%
- Context Precision: 60%
- Context Recall: 65%

### Step 2: Hybrid Retrieval Evaluation

Add BM25 + Ensemble retrieval:

```python
# Hybrid: BM25 + Dense with RRF
hybrid_results = evaluator.run_evaluation(
    retrieval_method="hybrid",
    bm25_weight=0.4,
    dense_weight=0.6,
    k=5,
    reranking=False
)
```

**Expected improvements**:
- Context Precision: +10-15%
- Context Recall: +10-15%

### Step 3: Reranking Evaluation

Add CrossEncoder reranking:

```python
# Hybrid + Reranking
reranked_results = evaluator.run_evaluation(
    retrieval_method="hybrid",
    bm25_weight=0.4,
    dense_weight=0.6,
    initial_k=20,
    final_k=5,
    reranking=True
)
```

**Expected improvements**:
- Context Precision: +15-20%
- Faithfulness: +10-15%

### Step 4: Query Rewriting Evaluation

Add LLM query expansion:

```python
# Full pipeline: Hybrid + Reranking + Query Rewriting
full_results = evaluator.run_evaluation(
    retrieval_method="hybrid",
    bm25_weight=0.4,
    dense_weight=0.6,
    initial_k=20,
    final_k=5,
    reranking=True,
    query_rewrite=True
)
```

**Expected improvements**:
- Answer Relevancy: +10-15%
- Context Recall: +5-10%

## A/B Testing Framework

### Chunking Strategy Comparison

| Strategy | Chunk Size | Overlap | Expected Impact |
|----------|-----------|---------|-----------------|
| Small | 500 | 100 | Higher precision, lower recall |
| Medium | 1000 | 200 | Balanced (default) |
| Large | 1500 | 300 | Higher recall, lower precision |

### Retrieval Weight Comparison

| Config | BM25 Weight | Dense Weight | Best For |
|--------|-------------|--------------|----------|
| Dense-heavy | 0.3 | 0.7 | Semantic queries |
| Balanced | 0.5 | 0.5 | General purpose |
| BM25-heavy | 0.7 | 0.3 | Keyword-heavy queries |

### Evaluation Script

```python
def run_ab_tests(evaluator, eval_dataset):
    """Run A/B tests on different configurations."""
    
    configs = {
        "baseline": {"retrieval": "dense", "k": 5, "rerank": False},
        "hybrid": {"retrieval": "hybrid", "k": 5, "rerank": False},
        "hybrid_rerank": {"retrieval": "hybrid", "initial_k": 20, "final_k": 5, "rerank": True},
        "full": {"retrieval": "hybrid", "initial_k": 20, "final_k": 5, "rerank": True, "rewrite": True}
    }
    
    results = {}
    for name, config in configs.items():
        results[name] = evaluator.run_evaluation(eval_dataset, **config)
    
    return results
```

## Failure Analysis

### Failure Categories

| Category | Description | Example |
|----------|-------------|---------|
| **Retrieval Miss** | Correct info not retrieved | Answer requires data from section not in top-k |
| **Retrieval Noise** | Irrelevant chunks retrieved | Retrieved chunks about different company |
| **Generation Hallucination** | Answer contains unsupported claims | Numbers not in context |
| **Incomplete Answer** | Answer missing key information | Partial answer without all required points |
| **Off-Topic Answer** | Answer addresses wrong question | Answer about revenue when asked about risks |

### Failure Documentation Template

```json
{
  "query": "What are Apple's main competitors?",
  "expected": "Apple competes with Samsung, Google, Microsoft, and other technology companies.",
  "actual": "Apple's main competitors include Samsung and Google.",
  "failure_type": "incomplete_answer",
  "root_cause": "Retrieved chunks only contained partial competitive landscape",
  "improvement": "Increase retrieval k or improve chunking to capture full sections"
}
```

### Common Failure Patterns

#### Pattern 1: Chunk Boundary Split
- **Symptom**: Important information split across chunks
- **Detection**: Answer incomplete despite info being in documents
- **Fix**: Increase chunk_overlap or use larger chunks

#### Pattern 2: Metadata Filter Over-Filtering
- **Symptom**: 40% retrieval drop after adding filters
- **Detection**: Context recall drops significantly
- **Fix**: Review filter logic, use more granular filters

#### Pattern 3: Reranking Latency
- **Symptom**: 200ms+ added latency
- **Detection**: Latency monitoring
- **Fix**: Acceptable for accuracy gain; optimize if >300ms

#### Pattern 4: LLM Override
- **Symptom**: Faithful context but wrong answer
- **Detection**: High context precision, low faithfulness
- **Fix**: Strengthen system prompt, reduce temperature

## Metrics Dashboard

### Real-time Metrics (Langfuse)

| Metric | Current | 7-day Avg | Trend |
|--------|---------|-----------|-------|
| Query Latency | 450ms | 480ms | ↓ improving |
| Faithfulness | 92% | 90% | ↑ improving |
| Context Precision | 87% | 85% | ↑ improving |
| Error Rate | 0.5% | 0.8% | ↓ improving |

### Evaluation Report (Weekly)

```markdown
# Weekly Evaluation Report - Week X

## Summary
- Total queries evaluated: 100
- Average faithfulness: 91%
- Average context precision: 86%

## Improvements This Week
- Added reranking: +12% context precision
- Query rewriting: +8% answer relevancy

## Failures Analyzed
- 5 retrieval misses (mostly edge cases)
- 3 generation hallucinations (all with vague context)

## Next Steps
- Tune BM25 weights for technical queries
- Add more ground truth for comparative questions
```

## Evaluation Cadence

| Activity | Frequency | Scope |
|----------|-----------|-------|
| Full RAGAS evaluation | Weekly | All 100+ QA pairs |
| Regression test | On PR | 30 QA pairs (subset) |
| Failure analysis | Weekly | All failures from production |
| A/B test | Bi-weekly | Compare configurations |
| Metric review | Daily | Dashboard check |

## Success Criteria

### MVP (Week 4)

- [ ] Faithfulness: 85%+ (baseline: 55%)
- [ ] Answer Relevancy: 80%+ (baseline: 65%)
- [ ] Context Precision: 80%+ (baseline: 60%)
- [ ] Latency: <600ms p95 (baseline: 2s)
- [ ] Hallucination Rate: <10% (baseline: 25%)

### Production Ready (Future)

- [ ] Faithfulness: 92%+
- [ ] Answer Relevancy: 88%+
- [ ] Context Precision: 87%+
- [ ] Latency: <500ms p95
- [ ] Hallucination Rate: <5%

## Tools and Libraries

| Tool | Purpose | Version |
|------|---------|---------|
| RAGAS | Evaluation framework | Latest |
| Ollama | Judge LLM | 0.5.7+ |
| Langfuse | Monitoring | Latest |
| Pandas | Data analysis | Latest |
| Matplotlib | Visualization | Latest |
| Jupyter | Notebooks | Latest |

## References

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS Metrics Guide](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [LangChain Evaluation](https://python.langchain.com/docs/guides/evaluation/)
