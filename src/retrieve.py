"""Hybrid retrieval with BM25 and Dense."""

from typing import List, Optional, Dict, Any

from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from .config import config
from .store import VectorStore


class HybridRetriever:
    """Combine BM25 and Dense retrieval."""
    
    def __init__(self, chunks: Optional[List[Document]] = None):
        self.vector_store = VectorStore()
        self.chunks = chunks
        self.bm25_retriever = None
        self.dense_retriever = None
        self.ensemble_retriever = None
        
        if chunks:
            self._build_retrievers(chunks)
    
    def _build_retrievers(self, chunks: List[Document]):
        """Build all retrievers."""
        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = config.retrieval.initial_k
        
        # Dense retriever
        self.dense_retriever = self.vector_store.get_langchain_retriever(
            k=config.retrieval.initial_k
        )
        
        # Ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=[config.retrieval.bm25_weight, config.retrieval.dense_weight]
        )
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve documents using hybrid search."""
        if self.ensemble_retriever:
            return self.ensemble_retriever.invoke(query)
        else:
            # Fallback to dense only
            return self.vector_store.search(query, k=k or config.retrieval.initial_k)
    
    def update_weights(self, bm25_weight: float, dense_weight: float):
        """Update retrieval weights."""
        if abs(bm25_weight + dense_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        
        if self.ensemble_retriever:
            self.ensemble_retriever.weights = [bm25_weight, dense_weight]
    
    def retrieve_bm25_only(self, query: str) -> List[Document]:
        """Retrieve using BM25 only (baseline)."""
        if self.bm25_retriever:
            return self.bm25_retriever.invoke(query)
        return []
    
    def retrieve_dense_only(self, query: str) -> List[Document]:
        """Retrieve using dense only (baseline)."""
        return self.vector_store.search(query, k=config.retrieval.initial_k)
