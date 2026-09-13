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


@dataclass
class RAGResponse:
    """RAG pipeline response."""
    answer: str
    sources: List[Dict]
    query_rewrite: Optional[str]
    latency_ms: float


class RAGPipeline:
    """Complete RAG pipeline with all components."""
    
    def __init__(self, strategy: str = "full"):
        """strategy: full | rerank | hybrid | baseline.

        full    = rewrite + hybrid + rerank
        rerank  = hybrid + rerank (no rewrite)
        hybrid  = hybrid only (no rewrite, no rerank)
        baseline = dense-only retrieval, no rewrite, no rerank
        """
        self.strategy = strategy
        self.query_rewriter = QueryRewriter()
        self.retriever = None
        self.reranker = Reranker()
        self.generator = AnswerGenerator()
        self.tracer = Tracer()
    
    def initialize(self, chunks: List[Document]):
        """Initialize with document chunks."""
        self.retriever = HybridRetriever(chunks)
    
    def query(self, user_query: str) -> RAGResponse:
        """Process a user query."""
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
            
            # Step 4: Generation
            answer = self.generator.generate(user_query, reranked)
            trace.set_attribute("answer_length", len(answer))
        
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
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query_rewrite=rewritten_queries[0] if rewritten_queries else None,
            latency_ms=latency_ms
        )
    
    def stream(self, user_query: str) -> Generator[str, None, None]:
        """Stream a response."""
        # Retrieve and rerank
        documents = self.retriever.retrieve(user_query)
        reranked = self.reranker.rerank(user_query, documents)
        
        # Stream generation
        yield from self.generator.stream(user_query, reranked)
