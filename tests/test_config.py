"""Tests for configuration."""

import pytest
from src.config import config


def test_config_initialization():
    """Test config can be initialized."""
    assert config.ollama.base_url == "http://localhost:11434"
    assert config.qdrant.port == 6333
    assert config.chunking.chunk_size == 1000
