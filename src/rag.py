"""Complete RAG pipeline."""

import time
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass

from langchain.schema import Document

from .config import config
from .query_rewrite import QueryRewriter
from .retrieve import HybridRetriever
from .rerank import Reranker
from .generate import AnswerGenerator
from .monitoring import Tracer
from .selfrag import SelfRag
from .guardrails import Guardrails


@dataclass
class RAGResponse:
    """RAG pipeline response."""
    answer: str
    sources: List[Dict]
    query_rewrite: Optional[str]
    latency_ms: float
    cached: bool = False


class RAGPipeline:
    """Complete RAG pipeline with all components."""
    
    def __init__(self, strategy: str = "full", tenant_id: Optional[str] = None):
        """strategy: full | rerank | hybrid | baseline.

        full    = rewrite + hybrid + rerank
        rerank  = hybrid + rerank (no rewrite)
        hybrid  = hybrid only (no rewrite, no rerank)
        baseline = dense-only retrieval, no rewrite, no rerank
        """
        self.strategy = strategy
        self.tenant_id = tenant_id or config.tenant.tenant_id
        self.query_rewriter = QueryRewriter()
        self.retriever = None
        self.reranker = Reranker()
        self.generator = AnswerGenerator()
        self.tracer = Tracer()
        self.selfrag = SelfRag()
        self.guardrails = Guardrails()
        self._cache = None
    
    def _get_cache(self):
        """Lazy semantic cache (avoids import cycle at module load)."""
        if self._cache is None:
            from .cache import SemanticCache
            self._cache = SemanticCache(tenant_id=self.tenant_id)
        return self._cache
    
    def initialize(self, chunks: List[Document]):
        """Initialize with document chunks."""
        for chunk in chunks:
            chunk.metadata["tenant_id"] = self.tenant_id
        self.retriever = HybridRetriever(chunks, tenant_id=self.tenant_id)
    
    def clear_cache(self):
        """Drop cached responses (call after reingest)."""
        self._get_cache().clear()
    
    def query(self, user_query: str) -> RAGResponse:
        """Process a user query."""
        # Input guardrail: block injections / PII / off-topic before any work.
        if self.guardrails.check_input(user_query):
            return RAGResponse(
                answer=self.guardrails.refusal,
                sources=[],
                query_rewrite=None,
                latency_ms=0.0,
            )
        
        # Semantic cache (bypasses the full pipeline on a near-identical query)
        cache = self._get_cache()
        if self.strategy != "baseline":
            cached = cache.lookup(user_query)
            if cached:
                return cached
        
        start_time = time.time()
        
        with self.tracer.trace("rag_query") as trace:
            # Step 1: Query rewriting (full strategy only)
            rewritten_queries = None
            queries = [user_query]
            if self.strategy == "full" and self.query_rewriter.should_rewrite(user_query):
                rewritten_queries = self.query_rewriter.rewrite(user_query)
                queries = queries + rewritten_queries
                trace.set_attribute("query_rewritten", True)
            
            # Step 2: Retrieval (baseline = dense-only, others = hybrid RRF)
            if self.strategy == "baseline":
                documents = self.retriever.retrieve_dense_only(user_query)
            else:
                documents = self.retriever.retrieve_multi(queries)
            trace.set_attribute("retrieval_count", len(documents))
            
            # Step 3: Reranking (full and rerank strategies)
            reranked = documents
            if self.strategy in ("full", "rerank"):
                reranked = self.reranker.rerank(user_query, documents)
                trace.set_attribute("reranked_count", len(reranked))
                
                # Step 3b: Self-RAG — if rerank confidence is weak, widen
                # retrieval once and re-rerank before committing to an answer.
                if self.selfrag.needs_expansion(reranked):
                    wider = self.retriever.retrieve_multi(
                        queries, k=self.selfrag.expand_top_k)
                    reranked = self.reranker.rerank(user_query, wider)
                    trace.set_attribute("selfrag_expanded", True)
                    trace.set_attribute("reranked_count", len(reranked))
                
                # Step 3c: Self-RAG — refuse rather than hallucinate on
                # context that never clears the confidence floor.
                if not self.selfrag.should_answer(reranked):
                    return RAGResponse(
                        answer="I don't have enough context to answer this "
                               "question confidently.",
                        sources=[],
                        query_rewrite=rewritten_queries[0] if rewritten_queries else None,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
            
            # Step 4: Generation
            answer = self.generator.generate(user_query, reranked)
            trace.set_attribute("answer_length", len(answer))
            
            # Output guardrail: refuse answers that leak PII or admit nothing.
            if self.guardrails.check_output(answer):
                answer = self.guardrails.refusal
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Format sources
        sources = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "section": doc.metadata.get("section", "general")
            }
            for doc in reranked
        ]
        
        response = RAGResponse(
            answer=answer,
            sources=sources,
            query_rewrite=rewritten_queries[0] if rewritten_queries else None,
            latency_ms=latency_ms
        )
        
        # Cache this generated response (skip baseline caching: it writes noise
        # for the degraded path and reingest clears the cache anyway)
        if self.strategy != "baseline":
            cache.store(user_query, response)
        
        return response
    
    def stream(self, user_query: str) -> Generator[str, None, None]:
        """Stream a response."""
        # Retrieve and rerank
        documents = self.retriever.retrieve(user_query)
        reranked = self.reranker.rerank(user_query, documents)
        
        # Stream generation
        yield from self.generator.stream(user_query, reranked)
