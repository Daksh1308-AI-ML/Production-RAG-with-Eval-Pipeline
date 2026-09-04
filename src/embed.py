"""Embedding generation with Ollama."""

from typing import List

from langchain_ollama import OllamaEmbeddings

from .config import config


class EmbeddingGenerator:
    """Generate embeddings using Ollama."""
    
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=config.ollama.embedding_model,
            base_url=config.ollama.base_url
        )
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        return self.embeddings.embed_query(query)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return self.embeddings.embed_documents(texts)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return config.qdrant.embedding_dim
