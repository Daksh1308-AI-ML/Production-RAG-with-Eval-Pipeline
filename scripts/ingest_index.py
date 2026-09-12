"""Index parsed SEC filings into Qdrant: load filings.json, chunk, recreate collection, upsert."""

import json
import time

from src.chunker import DocumentChunker
from src.config import config
from src.store import VectorStore


def main() -> None:
    print("loading processed filings...", flush=True)
    with open(config.processed_dir / "filings.json", encoding="utf-8") as f:
        filings = json.load(f)

    print("chunking...", flush=True)
    chunks = DocumentChunker().chunk_documents(filings)
    print(f"chunks: {len(chunks)}", flush=True)

    print("recreating collection...", flush=True)
    vs = VectorStore()
    vs.create_collection(recreate=True)

    batch = 200
    t0 = time.time()
    done = 0
    for i in range(0, len(chunks), batch):
        chunk_batch = chunks[i:i + batch]
        vs.upsert_documents(chunk_batch)
        done += len(chunk_batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed else 0
        print(f"progress: {done}/{len(chunks)} upserted "
              f"({elapsed:.0f}s, ~{rate:.1f}/s)", flush=True)

    print(f"DONE upserted {done} chunks in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()