"""Document chunking with configurable strategies."""

from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .config import config


class DocumentChunker:
    """Chunk documents for embedding."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=list(config.chunking.separators)
        )
    
    def chunk_document(self, doc: Dict) -> List[Document]:
        """Chunk a single document."""
        # Create LangChain Document
        lc_doc = Document(
            page_content=doc["content"],
            metadata=doc["metadata"]
        )
        
        # Split into chunks
        chunks = self.splitter.split_documents([lc_doc])
        
        # Add chunk indices
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def chunk_documents(self, docs: List[Dict]) -> List[Document]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Document]) -> Dict:
        """Get statistics about chunks."""
        sizes = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_size": sum(sizes) / len(sizes) if sizes else 0,
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0
        }
