"""Incremental ingestion: parse one or more SEC filing files and upsert without rebuilding.

Usage: python scripts/ingest_one.py <path...> [--tenant default]
"""

import argparse
import json
from pathlib import Path
from typing import List

from src.chunker import DocumentChunker
from src.config import config
from src.ingest import SECIngestor
from src.store import VectorStore


def ingest_files(paths: List[str], tenant: str = None) -> int:
    """Parse + chunk + incrementally upsert each filing file. Returns total chunks."""
    ingestor = SECIngestor()
    chunker = DocumentChunker()
    vs = VectorStore(tenant_id=tenant)

    total = 0
    for p in paths:
        path = Path(p)
        print(f"parsing {path}...", flush=True)
        filing = ingestor.parse_filing(path)
        chunks = chunker.chunk_documents([filing])
        print(f"  {len(chunks)} chunks -> upserting...", flush=True)
        vs.upsert_incremental(chunks)
        total += len(chunks)

    print(f"DONE upserted {total} chunks (incremental, collection preserved)", flush=True)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="filing file(s) to ingest")
    parser.add_argument("--tenant", default=None, help="tenant_id for the ingested chunks")
    args = parser.parse_args()
    ingest_files(args.paths, args.tenant)