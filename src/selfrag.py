"""Self-RAG: adaptive retrieval with a rerank-confidence gate.

Flow: retrieve candidates -> rerank -> if the best rerank score is below
`self_rag.min_confidence`, re-retrieve with a wider k and re-rerank once; if
the wider pass still can't clear `refuse_below`, refuse to answer rather than
hallucinate on weak context.
"""

from typing import List, Optional

from langchain.schema import Document

from .config import config


class SelfRag:
    """Confidence-gated adaptive retrieval."""

    def __init__(self, enabled: Optional[bool] = None,
                 min_confidence: Optional[float] = None,
                 refuse_below: Optional[float] = None,
                 expand_top_k: Optional[int] = None):
        self.enabled = config.self_rag.enabled if enabled is None else enabled
        self.min_confidence = (
            config.self_rag.min_confidence if min_confidence is None else min_confidence)
        self.refuse_below = (
            config.self_rag.refuse_below if refuse_below is None else refuse_below)
        self.expand_top_k = (
            config.self_rag.expand_top_k if expand_top_k is None else expand_top_k)

    @staticmethod
    def _best_score(documents: List[Document]) -> float:
        scores = [d.metadata.get("rerank_score") for d in documents
                  if d.metadata.get("rerank_score") is not None]
        return max(scores) if scores else 0.0

    def should_answer(self, reranked: List[Document]) -> bool:
        """True when rerank confidence clears the refuse floor."""
        if not self.enabled:
            return True
        return self._best_score(reranked) >= self.refuse_below

    def needs_expansion(self, reranked: List[Document]) -> bool:
        """True when rerank confidence is weak but not hopeless."""
        if not self.enabled:
            return False
        score = self._best_score(reranked)
        return self.refuse_below <= score < self.min_confidence