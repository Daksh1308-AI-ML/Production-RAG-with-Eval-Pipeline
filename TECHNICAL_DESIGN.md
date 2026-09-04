# Technical Design

## Project Structure

```
production-rag/
├── data/
│   ├── sec_filings/              # Raw SEC filings
│   │   ├── aapl/
│   │   ├── msft/
│   │   ├── googl/
│   │   ├── amzn/
│   │   └── nvda/
│   ├── processed/                # Cleaned text
│   └── evaluation/               # QA pairs for RAGAS
│       ├── eval_dataset.json
│       └── ground_truth.json
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── ingest.py                 # Document ingestion
│   ├── chunker.py                # Text chunking
│   ├── embed.py                  # Embedding generation
│   ├── store.py                  # Qdrant operations
│   ├── retrieve.py               # Hybrid retrieval
│   ├── rerank.py                 # CrossEncoder reranking
│   ├── query_rewrite.py          # Query expansion
│   ├── generate.py               # LLM generation
│   ├── rag.py                    # Full RAG pipeline
│   ├── eval.py                   # RAGAS evaluation
│   ├── monitoring.py             # Langfuse integration
│   └── utils.py                  # Helper functions
├── app/
│   └── streamlit_app.py          # Streamlit UI
├── tests/
│   ├── __init__.py
│   ├── test_ingest.py
│   ├── test_retrieve.py
│   ├── test_rag.py
│   └── test_eval.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_retrieval_comparison.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_failure_analysis.ipynb
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── .dockerignore
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── PROJECT_REQUIREMENTS.md
├── SYSTEM_ARCHITECTURE.md
├── TECHNICAL_DESIGN.md
├── RAG_EVALUATION.md
└── ROADMAP.md
```

## Module Specifications

### 1. config.py - Configuration Management

```python
"""Central configuration for all components."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class OllamaConfig:
    """Ollama server configuration."""
    base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "qwen2.5:7b"
    temperature: float = 0.0
    num_ctx: int = 8192
    keep_alive: str = "24h"


@dataclass
class QdrantConfig:
    """Qdrant vector store configuration."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "sec_filings"
    embedding_dim: int = 768  # nomic-embed-text dimensions


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: tuple = ("\n\n", "\n", ". ", " ", "")


@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration."""
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    initial_k: int = 20  # Candidates before reranking
    final_k: int = 5     # After reranking


@dataclass
class RerankerConfig:
    """Reranker configuration."""
    model_name: str = "BAAI/bge-reranker-base"
    top_n: int = 5


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    judge_model: str = "qwen2.5:7b"
    eval_dataset_path: str = "data/evaluation/eval_dataset.json"
    metrics: tuple = ("faithfulness", "answer_relevancy", "context_precision")


@dataclass
class MonitoringConfig:
    """Langfuse monitoring configuration."""
    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"
    enabled: bool = False


@dataclass
class AppConfig:
    """Main application configuration."""
    ollama: OllamaConfig = None
    qdrant: QdrantConfig = None
    chunking: ChunkingConfig = None
    retrieval: RetrievalConfig = None
    reranker: RerankerConfig = None
    eval: EvalConfig = None
    monitoring: MonitoringConfig = None
    
    def __post_init__(self):
        self.ollama = self.ollama or OllamaConfig()
        self.qdrant = self.qdrant or QdrantConfig()
        self.chunking = self.chunking or ChunkingConfig()
        self.retrieval = self.retrieval or RetrievalConfig()
        self.reranker = self.reranker or RerankerConfig()
        self.eval = self.eval or EvalConfig()
        self.monitoring = self.monitoring or MonitoringConfig()


# Global config instance
config = AppConfig()
```

### 2. ingest.py - Document Ingestion

```python
"""SEC 10-K filing ingestion pipeline."""

from pathlib import Path
from typing import List, Dict
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
import re


class SECIngestor:
    """Download and parse SEC 10-K filings."""
    
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    FILING_TYPE = "10-K"
    FILING_LIMIT = 3  # Per company
    
    def __init__(self, data_dir: str = "data/sec_filings"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = Downloader(
            company_name="ResearchProject",
            email="research@example.com",
            download_folder=str(self.data_dir)
        )
    
    def download_all(self) -> Dict[str, int]:
        """Download 10-K filings for all tickers."""
        results = {}
        for ticker in self.TICKERS:
            count = self._download_ticker(ticker)
            results[ticker] = count
        return results
    
    def _download_ticker(self, ticker: str) -> int:
        """Download filings for a single ticker."""
        try:
            self.downloader.get(
                self.FILING_TYPE,
                ticker,
                limit=self.FILING_LIMIT
            )
            return self.FILING_LIMIT
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            return 0
    
    def parse_filing(self, filepath: Path) -> Dict:
        """Parse a single filing HTML file."""
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        
        # Remove scripts and styles
        for tag in soup(["script", "style"]):
            tag.decompose()
        
        # Extract text
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean text
        text = self._clean_text(text)
        
        # Extract metadata
        metadata = self._extract_metadata(filepath, text)
        
        return {
            "content": text,
            "metadata": metadata,
            "source": str(filepath)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Remove common headers/footers
        text = re.sub(r'(?i)table of contents.*?(?=\n\n)', '', text)
        
        return text.strip()
    
    def _extract_metadata(self, filepath: Path, text: str) -> Dict:
        """Extract metadata from filing."""
        # Parse ticker from path
        ticker = filepath.parent.name.upper()
        
        # Extract filing date (simplified)
        date_match = re.search(r' Filed: (\d{4}-\d{2}-\d{2})', text)
        filing_date = date_match.group(1) if date_match else "unknown"
        
        # Extract section (simplified)
        section = "general"
        if "risk factors" in text.lower()[:5000]:
            section = "risk_factors"
        elif "financial statements" in text.lower()[:5000]:
            section = "financial_statements"
        elif "business" in text.lower()[:5000]:
            section = "business"
        
        return {
            "ticker": ticker,
            "filing_type": "10-K",
            "filing_date": filing_date,
            "section": section,
            "source": str(filepath)
        }
```

### 3. chunker.py - Text Chunking

```python
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
```

### 4. embed.py - Embedding Generation

```python
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
```

### 5. store.py - Qdrant Operations

```python
"""Qdrant vector store operations."""

from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from .config import config
from .embed import EmbeddingGenerator
import uuid


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
        points = []
        for chunk in chunks:
            # Generate embedding
            embedding = self.embeddings.embed_query(chunk.page_content)
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                }
            )
            points.append(point)
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        return len(points)
    
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
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=query_filter
        )
        
        # Convert to Documents
        documents = []
        for result in results:
            doc = Document(
                page_content=result.payload["content"],
                metadata=result.payload["metadata"]
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
```

### 6. retrieve.py - Hybrid Retrieval

```python
"""Hybrid retrieval with BM25 and Dense."""

from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from .config import config
from .store import VectorStore


class HybridRetriever:
    """Combine BM25 and Dense retrieval."""
    
    def __init__(self, chunks: Optional[List[Document]] = None):
        self.vector_store = VectorStore()
        self.chunks = chunks
        self.bm25_retriever = None
        self.dense_retriever = None
        self.ensemble_retriever = None
        
        if chunks:
            self._build_retrievers(chunks)
    
    def _build_retrievers(self, chunks: List[Document]):
        """Build all retrievers."""
        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = config.retrieval.initial_k
        
        # Dense retriever
        self.dense_retriever = self.vector_store.get_langchain_retriever(
            k=config.retrieval.initial_k
        )
        
        # Ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=[config.retrieval.bm25_weight, config.retrieval.dense_weight]
        )
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve documents using hybrid search."""
        if self.ensemble_retriever:
            return self.ensemble_retriever.invoke(query)
        else:
            # Fallback to dense only
            return self.vector_store.search(query, k=k or config.retrieval.initial_k)
    
    def update_weights(self, bm25_weight: float, dense_weight: float):
        """Update retrieval weights."""
        if abs(bm25_weight + dense_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        
        if self.ensemble_retriever:
            self.ensemble_retriever.weights = [bm25_weight, dense_weight]
    
    def retrieve_bm25_only(self, query: str) -> List[Document]:
        """Retrieve using BM25 only (baseline)."""
        if self.bm25_retriever:
            return self.bm25_retriever.invoke(query)
        return []
    
    def retrieve_dense_only(self, query: str) -> List[Document]:
        """Retrieve using dense only (baseline)."""
        return self.vector_store.search(query, k=config.retrieval.initial_k)
```

### 7. rerank.py - CrossEncoder Reranking

```python
"""Reranking with CrossEncoder model."""

from typing import List
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from .config import config


class Reranker:
    """Rerank retrieved documents."""
    
    def __init__(self):
        self.cross_encoder = HuggingFaceCrossEncoder(
            model_name=config.reranker.model_name
        )
        self.compressor = CrossEncoderReranker(
            model=self.cross_encoder,
            top_n=config.reranker.top_n
        )
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = None
    ) -> List[Document]:
        """Rerank documents by relevance."""
        top_n = top_n or config.reranker.top_n
        
        # Score and sort
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.cross_encoder.score(pairs)
        
        # Combine and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_n
        return [doc for doc, score in doc_scores[:top_n]]
    
    def get_compression_retriever(self, base_retriever):
        """Get a LangChain compression retriever."""
        return ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever
        )
```

### 8. query_rewrite.py - Query Rewriting

```python
"""Query rewriting and expansion."""

from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from .config import config


class QueryRewriter:
    """Rewrite and expand queries."""
    
    REWRITE_PROMPT = """You are a query rewriting assistant.
Given the user's question, generate 3-5 alternative search queries that would help
find relevant information. Focus on different phrasings and keywords.

Original question: {query}

Output only the rewritten queries, one per line:"""
    
    EXPAND_PROMPT = """You are a query expansion assistant.
Given the user's question, expand it with relevant context and synonyms.
Make the query more specific and searchable.

Original question: {query}

Expanded query:"""
    
    def __init__(self):
        self.llm = ChatOllama(
            model=config.ollama.llm_model,
            temperature=0.3,
            base_url=config.ollama.base_url
        )
    
    def rewrite(self, query: str) -> List[str]:
        """Generate multiple search queries."""
        prompt = ChatPromptTemplate.from_template(self.REWRITE_PROMPT)
        chain = prompt | self.llm
        
        response = chain.invoke({"query": query})
        
        # Parse response
        queries = [q.strip() for q in response.content.split("\n") if q.strip()]
        return queries[:5]
    
    def expand(self, query: str) -> str:
        """Expand a single query."""
        prompt = ChatPromptTemplate.from_template(self.EXPAND_PROMPT)
        chain = prompt | self.llm
        
        response = chain.invoke({"query": query})
        return response.content.strip()
    
    def should_rewrite(self, query: str) -> bool:
        """Determine if query needs rewriting."""
        # Rewrite if query is short or ambiguous
        words = query.split()
        if len(words) < 5:
            return True
        if "?" not in query and len(words) < 10:
            return True
        return False
```

### 9. generate.py - LLM Generation

```python
"""LLM answer generation with citations."""

from typing import List, Dict, Generator
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .config import config


class AnswerGenerator:
    """Generate answers with source citations."""
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions about SEC 10-K filings.
Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer this question."
Always cite your sources by referencing the document section."""
    
    CONTEXT_PROMPT = """Context:
{context}

Question: {question}

Answer (include citations to source documents):"""
    
    def __init__(self):
        self.llm = ChatOllama(
            model=config.ollama.llm_model,
            temperature=config.ollama.temperature,
            num_ctx=config.ollama.num_ctx,
            base_url=config.ollama.base_url
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.CONTEXT_PROMPT)
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            section = doc.metadata.get("section", "general")
            context_parts.append(
                f"[Source {i}: {source} - {section}]\n{doc.page_content}"
            )
        return "\n\n".join(context_parts)
    
    def generate(self, query: str, documents: List[Document]) -> str:
        """Generate an answer."""
        context = self._format_context(documents)
        
        response = self.chain.invoke({
            "context": context,
            "question": query
        })
        
        return response
    
    def stream(self, query: str, documents: List[Document]) -> Generator[str, None, None]:
        """Stream an answer."""
        context = self._format_context(documents)
        
        for chunk in self.chain.stream({
            "context": context,
            "question": query
        }):
            yield chunk
```

### 10. rag.py - Full RAG Pipeline

```python
"""Complete RAG pipeline."""

from typing import List, Dict, Optional, Generator
from dataclasses import dataclass
from langchain.schema import Document
from .config import config
from .query_rewrite import QueryRewriter
from .retrieve import HybridRetriever
from .rerank import Reranker
from .generate import AnswerGenerator
from .monitoring import Tracer


@dataclass
class RAGResponse:
    """RAG pipeline response."""
    answer: str
    sources: List[Dict]
    query_rewrite: Optional[str]
    latency_ms: float


class RAGPipeline:
    """Complete RAG pipeline with all components."""
    
    def __init__(self):
        self.query_rewriter = QueryRewriter()
        self.retriever = None
        self.reranker = Reranker()
        self.generator = AnswerGenerator()
        self.tracer = Tracer()
    
    def initialize(self, chunks: List[Document]):
        """Initialize with document chunks."""
        self.retriever = HybridRetriever(chunks)
    
    def query(self, user_query: str) -> RAGResponse:
        """Process a user query."""
        import time
        start_time = time.time()
        
        with self.tracer.trace("rag_query") as trace:
            # Step 1: Query rewriting
            rewritten_queries = None
            if self.query_rewriter.should_rewrite(user_query):
                rewritten_queries = self.query_rewriter.rewrite(user_query)
                trace.set_attribute("query_rewritten", True)
            
            # Step 2: Retrieval
            documents = self.retriever.retrieve(user_query)
            trace.set_attribute("retrieval_count", len(documents))
            
            # Step 3: Reranking
            reranked = self.reranker.rerank(user_query, documents)
            trace.set_attribute("reranked_count", len(reranked))
            
            # Step 4: Generation
            answer = self.generator.generate(user_query, reranked)
            trace.set_attribute("answer_length", len(answer))
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Format sources
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "Unknown"),
                "section": doc.metadata.get("section", "general")
            }
            for doc in reranked
        ]
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query_rewrite=rewritten_queries[0] if rewritten_queries else None,
            latency_ms=latency_ms
        )
    
    def stream(self, user_query: str) -> Generator[str, None, None]:
        """Stream a response."""
        # Retrieve and rerank
        documents = self.retriever.retrieve(user_query)
        reranked = self.reranker.rerank(user_query, documents)
        
        # Stream generation
        yield from self.generator.stream(user_query, reranked)
```

### 11. eval.py - RAGAS Evaluation

```python
"""RAGAS evaluation pipeline."""

import json
from typing import List, Dict
from pathlib import Path
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.dataset_schema import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from .config import config


class RAGEvaluator:
    """Evaluate RAG system with RAGAS."""
    
    def __init__(self):
        self.judge_llm = ChatOllama(
            model=config.eval.judge_model,
            temperature=0,
            base_url=config.ollama.base_url
        )
        self.judge_embeddings = OllamaEmbeddings(
            model=config.ollama.embedding_model,
            base_url=config.ollama.base_url
        )
    
    def load_eval_dataset(self) -> Dataset:
        """Load evaluation dataset."""
        dataset_path = Path(config.eval.eval_dataset_path)
        
        with open(dataset_path, "r") as f:
            data = json.load(f)
        
        return Dataset.from_dict(data)
    
    def run_evaluation(
        self,
        rag_pipeline,
        eval_dataset: Dataset = None
    ) -> Dict:
        """Run full RAGAS evaluation."""
        if eval_dataset is None:
            eval_dataset = self.load_eval_dataset()
        
        # Generate answers for each question
        generated_data = []
        for item in eval_dataset:
            response = rag_pipeline.query(item["question"])
            generated_data.append({
                "question": item["question"],
                "contexts": [s["content"] for s in response.sources],
                "answer": response.answer,
                "ground_truth": item.get("ground_truth", "")
            })
        
        # Create evaluation dataset
        eval_ds = Dataset.from_dict({
            "question": [d["question"] for d in generated_data],
            "contexts": [d["contexts"] for d in generated_data],
            "answer": [d["answer"] for d in generated_data],
            "ground_truth": [d["ground_truth"] for d in generated_data]
        })
        
        # Run RAGAS evaluation
        result = evaluate(
            dataset=eval_ds,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=self.judge_llm,
            embeddings=self.judge_embeddings
        )
        
        return {
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
            "context_precision": result["context_precision"],
            "context_recall": result["context_recall"],
            "details": result
        }
    
    def compare_strategies(
        self,
        pipelines: Dict[str, any],
        eval_dataset: Dataset = None
    ) -> Dict:
        """Compare multiple RAG strategies."""
        results = {}
        for name, pipeline in pipelines.items():
            results[name] = self.run_evaluation(pipeline, eval_dataset)
        return results
```

### 12. monitoring.py - Langfuse Integration

```python
"""Langfuse monitoring and tracing."""

from typing import Optional, Dict, Any
from contextlib import contextmanager
from .config import config


class Tracer:
    """Langfuse tracer for monitoring."""
    
    def __init__(self):
        self.client = None
        if config.monitoring.enabled:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(
                    public_key=config.monitoring.public_key,
                    secret_key=config.monitoring.secret_key,
                    host=config.monitoring.host
                )
            except Exception as e:
                print(f"Langfuse initialization failed: {e}")
    
    @contextmanager
    def trace(self, name: str, metadata: Optional[Dict] = None):
        """Context manager for tracing."""
        if not self.client:
            yield NoOpTrace()
            return
        
        trace = self.client.trace(name=name, metadata=metadata or {})
        try:
            yield trace
        except Exception as e:
            trace.span(name="error", input={"error": str(e)})
            raise
        finally:
            self.client.flush()
    
    def score(self, trace_id: str, name: str, value: float):
        """Record a score."""
        if self.client:
            self.client.score(trace_id=trace_id, name=name, value=value)


class NoOpTrace:
    """No-op trace when monitoring is disabled."""
    
    def set_attribute(self, key: str, value: Any):
        pass
    
    def span(self, **kwargs):
        pass
```

## Configuration Files

### .env.example

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=qwen2.5:7b

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=sec_filings

# Langfuse Configuration (optional)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=false
```

### requirements.txt

```
langchain==0.3.18
langchain-community==0.3.18
langchain-ollama==0.2.3
langchain-core==0.3.34
langchain-qdrant
qdrant-client==1.13.0
ragas
streamlit
sec-edgar-downloader
rank-bm25
sentence-transformers
langfuse
python-dotenv
beautifulsoup4
pypdf
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.13.0
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

  ollama:
    image: ollama/ollama
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  qdrant_data:
  ollama_data:
```

## Error Handling

### Error Types

```python
class RAGError(Exception):
    """Base RAG error."""
    pass

class IngestionError(RAGError):
    """Document ingestion error."""
    pass

class RetrievalError(RAGError):
    """Retrieval error."""
    pass

class GenerationError(RAGError):
    """LLM generation error."""
    pass

class EvaluationError(RAGError):
    """Evaluation error."""
    pass
```

### Recovery Strategies

| Error Type | Strategy |
|-----------|----------|
| Ollama Down | Retry with exponential backoff |
| Qdrant Down | Fall back to in-memory |
| Embedding Mismatch | Rebuild index |
| LLM Timeout | Use cached response |
| Memory OOM | Reduce batch size |
