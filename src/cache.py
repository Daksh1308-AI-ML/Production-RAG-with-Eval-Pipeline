"""Semantic cache: store query->answer points in Qdrant, serve on cosine >= threshold."""

from __future__ import annotations

from typing import Optional
from uuid import uuid5, NAMESPACE_DNS

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
)

from .config import config
from .embed import EmbeddingGenerator
from .rag import RAGResponse


class SemanticCache:
    """Embed query, top-1 cosine >= threshold -> cached response; else miss.

    Stores a single point per (tenant_id, query) — one entry per distinct
    question, so the collection stays small (a handful of points, not per-chunk)."""

    def __init__(self, tenant_id: Optional[str] = None):
        self.collection = config.cache.collection_name
        self.threshold = config.cache.threshold
        self.tenant_id = tenant_id or config.tenant.tenant_id
        self.client = QdrantClient(host=config.qdrant.host, port=config.qdrant.port)
        self.embeddings = EmbeddingGenerator()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=config.qdrant.embedding_dim,
                    distance=Distance.COSINE
                )
            )

    @staticmethod
    def _point_id(query: str, tenant_id: str) -> str:
        return str(uuid5(NAMESPACE_DNS, f"{tenant_id}|{query}"))

    def lookup(self, query: str) -> Optional[RAGResponse]:
        """Return cached response if a similar query (>= threshold) exists."""
        if not config.cache.enabled:
            return None
        self._ensure_collection()
        hits = self.client.query_points(
            collection_name=self.collection,
            query=self.embeddings.embed_query(query),
            limit=1,
            query_filter=Filter(must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=self.tenant_id))
            ]),
            score_threshold=self.threshold
        ).points
        if not hits:
            return None
        payload = hits[0].payload
        return RAGResponse(
            answer=payload["answer"],
            sources=payload["sources"],
            query_rewrite=payload.get("query_rewrite"),
            latency_ms=0.0,
            cached=True,
        )

    def store(self, query: str, response: RAGResponse):
        """Cache a generated response."""
        if not config.cache.enabled:
            return
        self._ensure_collection()
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=self._point_id(query, self.tenant_id),
                vector=self.embeddings.embed_query(query),
                payload={
                    "tenant_id": self.tenant_id,
                    "answer": response.answer,
                    "sources": response.sources,
                    "query_rewrite": response.query_rewrite,
                }
            )]
        )

    def clear(self):
        """Drop the cache (call after reingest)."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)