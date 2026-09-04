"""Helper functions."""

import json
from pathlib import Path
from typing import List, Dict, Any

from langchain.schema import Document


def save_documents_to_json(documents: List[Document], filepath: str):
    """Save documents to JSON file."""
    data = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_documents_from_json(filepath: str) -> List[Document]:
    """Load documents from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    return [
        Document(
            page_content=item["content"],
            metadata=item["metadata"]
        )
        for item in data
    ]


def format_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    """Format documents for display."""
    return [
        {
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "section": doc.metadata.get("section", "general"),
            "ticker": doc.metadata.get("ticker", "Unknown"),
            "chunk_index": doc.metadata.get("chunk_index", 0)
        }
        for doc in documents
    ]


def print_retrieval_results(documents: List[Document], query: str):
    """Print retrieval results for debugging."""
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(documents)} documents:")
    for i, doc in enumerate(documents, 1):
        ticker = doc.metadata.get("ticker", "Unknown")
        section = doc.metadata.get("section", "general")
        print(f"  {i}. [{ticker}] {section} - {doc.page_content[:100]}...")
