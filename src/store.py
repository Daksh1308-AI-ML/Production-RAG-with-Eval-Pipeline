"""Qdrant vector store operations."""

from typing import List, Optional, Dict, Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore

from .config import config
from .embed import EmbeddingGenerator


class VectorStore:
    """Qdrant vector store manager."""
    
    def __init__(self):
        self.client = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port
        )
        self.embeddings = EmbeddingGenerator()
        self.collection_name = config.qdrant.collection_name
    
    def create_collection(self, recreate: bool = False):
        """Create or recreate the collection."""
        if recreate:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=config.qdrant.embedding_dim,
                distance=Distance.COSINE
            )
        )
    
    def upsert_documents(self, chunks: List[Document]) -> int:
        """Upsert document chunks to Qdrant."""
        # Embed in batches (one HTTP call per batch instead of per chunk)
        batch_size = 200
        points = []
        for i in range(0, len(chunks), batch_size):
            batch_docs = chunks[i:i + batch_size]
            embeddings = self.embeddings.embed_documents(
                [d.page_content for d in batch_docs]
            )
            for chunk, embedding in zip(batch_docs, embeddings):
                points.append(PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload={
                        "content": chunk.page_content,
                        "metadata": chunk.metadata
                    }
                ))

            # Upsert in batches
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                points = []

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        return len(chunks)
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        
        # Build filter
        query_filter = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)
        
        # Search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,
            query_filter=query_filter
        ).points
        
        # Convert to Documents
        documents = []
        for result in results:
            metadata = dict(result.payload.get("metadata", {}))
            metadata["score"] = result.score
            doc = Document(
                page_content=result.payload["content"],
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def get_langchain_retriever(self, k: int = 5):
        """Get a LangChain retriever interface."""
        vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings.embeddings
        )
        return vectorstore.as_retriever(search_kwargs={"k": k})
