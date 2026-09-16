"""FastAPI gateway for the RAG pipeline.

Endpoints:
  POST /query   {query, strategy?, tenant_id?} -> RAGResponse
  POST /ingest  {paths: [...]}                 -> chunk counts
  GET  /health                                 -> service + collection status

Auth: X-API-Key header must match one of the comma-separated API_KEYS.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.chunker import DocumentChunker
from src.rag import RAGPipeline

app = FastAPI(title="Production RAG API", version="2.0.0")

STRATEGIES = ("full", "rerank", "hybrid", "baseline")


class QueryRequest(BaseModel):
    query: str
    strategy: str = "full"
    tenant_id: Optional[str] = None


class IngestRequest(BaseModel):
    paths: List[str]


def _require_key(x_api_key: Optional[str] = Header(None)) -> None:
    if not config.api.api_keys or x_api_key not in config.api.api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _load_pipeline(strategy: str, tenant_id: Optional[str]) -> RAGPipeline:
    """One pipeline per (strategy, tenant), built lazily and cached."""
    key = f"{strategy}|{tenant_id or config.tenant.tenant_id}"
    if key not in app.state.pipelines:
        filings = json.load(open(config.processed_dir / "filings.json", encoding="utf-8"))
        chunks = DocumentChunker().chunk_documents(filings)
        pipeline = RAGPipeline(strategy=strategy, tenant_id=tenant_id)
        pipeline.initialize(chunks)
        app.state.pipelines[key] = pipeline
    return app.state.pipelines[key]


@app.on_event("startup")
def _startup():
    app.state.pipelines = {}


@app.post("/query")
def query(req: QueryRequest, x_api_key: Optional[str] = Header(None)):
    _require_key(x_api_key)
    if req.strategy not in STRATEGIES:
        raise HTTPException(status_code=422, detail=f"strategy must be one of {STRATEGIES}")
    pipeline = _load_pipeline(req.strategy, req.tenant_id)
    try:
        response = pipeline.query(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "answer": response.answer,
        "sources": response.sources,
        "query_rewrite": response.query_rewrite,
        "latency_ms": response.latency_ms,
        "cached": response.cached,
    }


@app.post("/ingest")
def ingest(req: IngestRequest, x_api_key: Optional[str] = Header(None)):
    _require_key(x_api_key)
    from scripts.ingest_one import ingest_files

    total = ingest_files(req.paths, tenant=config.tenant.tenant_id)
    # New source chunks may have changed, so drop stale cached answers.
    for pipeline in app.state.pipelines.values():
        pipeline.clear_cache()
    return {"ingested_chunks": total}


@app.get("/health")
def health():
    qdrant_ok = ollama_ok = False
    qdrant_points = None
    try:
        from src.store import VectorStore
        vs = VectorStore()
        qdrant_ok = vs.client.collection_exists(vs.collection_name)
        if qdrant_ok:
            qdrant_points = vs.client.count(vs.collection_name).count
    except Exception:
        pass
    try:
        import requests
        ollama_ok = requests.get(
            f"{config.ollama.base_url}/api/tags", timeout=3
        ).status_code == 200
    except Exception:
        pass
    return {
        "status": "ok" if qdrant_ok and ollama_ok else "degraded",
        "qdrant": qdrant_ok,
        "qdrant_points": qdrant_points,
        "ollama": ollama_ok,
        "collections": {
            "sec_filings": config.qdrant.collection_name,
            "semantic_cache": config.cache.collection_name,
        },
    }