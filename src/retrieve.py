"""Hybrid retrieval with BM25 and Dense, fused via Reciprocal Rank Fusion."""

from typing import List, Optional, Dict, Any

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever

from .config import config
from .store import VectorStore

# ponytail: manual RRF replaces langchain EnsembleRetriever — the wrapper
# silently returned < k results and ignored filter_metadata on the ensemble path.
# RRF with config weights is 8 lines; upgrade path if behavior drifts: swap in langchain tools.
_RRF_C = 60


class HybridRetriever:
    """Combine BM25 and Dense retrieval."""
    
    def __init__(self, chunks: Optional[List[Document]] = None,
                 tenant_id: Optional[str] = None):
        self.vector_store = VectorStore(tenant_id=tenant_id)
        self.chunks = chunks
        self.tenant_id = tenant_id or config.tenant.tenant_id
        self.bm25_retriever = None
        self.dense_retriever = None
        
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
    
    def _fusion_key(self, doc: Document) -> str:
        """Deduplication key for RRF scoring."""
        ticker = doc.metadata.get("ticker")
        return f"{ticker}|{doc.page_content}" if ticker else doc.page_content
    
    def _rrf(self, queries: List[str], k: int) -> List[Document]:
        """Fuse BM25 + dense results for one or more queries via weighted RRF."""
        scores: Dict[str, List[object]] = {}
        
        for query in queries:
            bm25 = self.retrieve_bm25_only(query, k=k)
            dense = self.retrieve_dense_only(query, k=k)
            for rank, doc in enumerate(bm25):
                key = self._fusion_key(doc)
                scores.setdefault(key, [0.0, doc])[0] += (
                    config.retrieval.bm25_weight / (rank + _RRF_C))
            for rank, doc in enumerate(dense):
                key = self._fusion_key(doc)
                scores.setdefault(key, [0.0, doc])[0] += (
                    config.retrieval.dense_weight / (rank + _RRF_C))
        
        ranked = sorted(scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked]
    
    def _filter(self, documents: List[Document],
                filter_metadata: Optional[Dict[str, Any]]) -> List[Document]:
        """Apply tenant + metadata filter post-hoc (BM25 can't filter natively)."""
        return [
            d for d in documents
            if d.metadata.get("tenant_id", self.tenant_id) == self.tenant_id
            and all(d.metadata.get(key) == value
                    for key, value in (filter_metadata or {}).items())
        ]
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve documents using hybrid search."""
        documents = self.retrieve_multi([query], k=k,
                                        filter_metadata=filter_metadata)
        return documents
    
    def retrieve_multi(
        self,
        queries: List[str],
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve with multiple queries (query expansion), fused via RRF."""
        k = k or config.retrieval.initial_k
        documents = self._filter(self._rrf(queries, k), filter_metadata)
        return documents[:k]
    
    def update_weights(self, bm25_weight: float, dense_weight: float):
        """Update retrieval weights."""
        if abs(bm25_weight + dense_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        config.retrieval.bm25_weight = bm25_weight
        config.retrieval.dense_weight = dense_weight
    
    def retrieve_bm25_only(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve using BM25 only (baseline)."""
        if self.bm25_retriever:
            self.bm25_retriever.k = k or config.retrieval.initial_k
            return self.bm25_retriever.invoke(query)
        return []
    
    def retrieve_dense_only(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve using dense only (baseline)."""
        return self.vector_store.search(query, k=k or config.retrieval.initial_k)
