"""Tests for document ingestion."""

import pytest
from src.ingest import SECIngestor


def test_ingestor_initialization():
    """Test SECIngestor can be initialized."""
    ingestor = SECIngestor()
    assert ingestor.data_dir.exists()
