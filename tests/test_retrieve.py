"""Integration tests for dense retrieval against the live Qdrant corpus.

Skipped when Qdrant or Ollama is not reachable.
"""

import pytest

from src.store import VectorStore


def _collection_info() -> int:
    """Return point count of the sec_filings collection, or raise."""
    vs = VectorStore()
    info = vs.client.get_collection(vs.collection_name)
    return info.points_count


@pytest.fixture(scope="module")
def vector_store():
    vs = VectorStore()
    try:
        count = _collection_info()
    except Exception as e:
        pytest.skip(f"Qdrant unavailable: {e}")
    if not count:
        pytest.skip("collection empty")
    return vs


def test_collection_populated(vector_store):
    count = _collection_info()
    assert count == 39565


def test_dense_search_returns_k_docs(vector_store):
    docs = vector_store.search("What were Apple's net sales?", k=5)
    assert len(docs) == 5
    assert all(d.metadata.get("ticker") for d in docs)
    assert all(d.metadata.get("score") is not None for d in docs)


def test_metadata_filter(vector_store):
    docs = vector_store.search("operating income", k=5,
                               filter_metadata={"ticker": "AAPL"})
    assert len(docs) == 5
    assert all(d.metadata.get("ticker") == "AAPL" for d in docs)