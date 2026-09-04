"""Central configuration for all components."""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class OllamaConfig:
    """Ollama server configuration."""
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b")
    temperature: float = 0.0
    num_ctx: int = 8192
    keep_alive: str = "24h"


@dataclass
class QdrantConfig:
    """Qdrant vector store configuration."""
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = os.getenv("QDRANT_COLLECTION", "sec_filings")
    embedding_dim: int = int(os.getenv("QDRANT_EMBEDDING_DIM", "768"))


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    separators: Tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration."""
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.4"))
    dense_weight: float = float(os.getenv("DENSE_WEIGHT", "0.6"))
    initial_k: int = int(os.getenv("INITIAL_K", "20"))
    final_k: int = int(os.getenv("FINAL_K", "5"))


@dataclass
class RerankerConfig:
    """Reranker configuration."""
    model_name: str = "BAAI/bge-reranker-base"
    top_n: int = int(os.getenv("FINAL_K", "5"))


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    judge_model: str = os.getenv("JUDGE_MODEL", "qwen2.5:7b")
    eval_dataset_path: str = os.getenv("EVAL_DATASET_PATH", "data/evaluation/eval_dataset.json")
    metrics: Tuple[str, ...] = ("faithfulness", "answer_relevancy", "context_precision")


@dataclass
class MonitoringConfig:
    """Langfuse monitoring configuration."""
    public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    enabled: bool = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"


@dataclass
class SECConfig:
    """SEC EDGAR configuration."""
    company_name: str = os.getenv("SEC_COMPANY_NAME", "ResearchProject")
    email: str = os.getenv("SEC_EMAIL", "research@example.com")
    tickers: Tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA")
    filing_type: str = "10-K"
    filing_limit: int = 3


@dataclass
class AppConfig:
    """Main application configuration."""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    sec: SECConfig = field(default_factory=SECConfig)
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    sec_filings_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "sec_filings")
    processed_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "processed")
    evaluation_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "evaluation")


# Global config instance
config = AppConfig()
