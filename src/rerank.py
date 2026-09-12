"""Reranking with CrossEncoder model."""

from typing import List

from langchain.schema import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from .config import config


class Reranker:
    """Rerank retrieved documents."""
    
    def __init__(self, model_name: str = None):
        self.cross_encoder = HuggingFaceCrossEncoder(
            model_name=model_name or config.reranker.model_name
        )
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = None
    ) -> List[Document]:
        """Rerank documents by relevance."""
        top_n = top_n or config.reranker.top_n
        
        if not documents:
            return []
        
        # Score and sort
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.cross_encoder.score(pairs)
        
        # Combine and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_n
        return [doc for doc, score in doc_scores[:top_n]]
