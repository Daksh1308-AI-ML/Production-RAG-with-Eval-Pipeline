"""Week 2 validation: hybrid retrieval (BM25 + dense RRF) and reranking.

Run with `--no-rerank` to skip the reranker (avoids the ~1GB model download).
Use `--reranker-model NAME` to override the default (smaller/faster cross-encoders).
"""

import json
import sys
import time

from src.chunker import DocumentChunker
from src.config import config
from src.retrieve import HybridRetriever

USE_RERANK = "--no-rerank" not in sys.argv
_RERANKER_MODEL = None
if "--reranker-model" in sys.argv:
    _RERANKER_MODEL = sys.argv[sys.argv.index("--reranker-model") + 1]

SAMPLE_QUERIES = [
    ("AAPL", "What were Apple's total net sales in fiscal 2024?"),
    ("MSFT", "How much revenue did Microsoft cloud generate?"),
    ("GOOGL", "What is Alphabet's total advertising revenue?"),
    ("AMZN", "What operating segments does Amazon report?"),
    ("NVDA", "What was NVIDIA's data center revenue?"),
    (None, "What supply chain and manufacturing risk factors do these companies disclose?"),
]


def _tickers(docs):
    return [d.metadata.get("ticker") for d in docs]


def main() -> int:
    print("loading filings.json...", flush=True)
    with open(config.processed_dir / "filings.json", encoding="utf-8") as f:
        filings = json.load(f)

    print("chunking for BM25 index...", flush=True)
    chunks = DocumentChunker().chunk_documents(filings)
    print(f"chunks: {len(chunks)}", flush=True)

    print("building hybrid retriever (BM25 + dense)...", flush=True)
    t0 = time.time()
    hr = HybridRetriever(chunks)
    print(f"built in {time.time() - t0:.1f}s", flush=True)

    reranker = None
    if USE_RERANK:
        print("loading reranker (bge-reranker-base)...", flush=True)
        t0 = time.time()
        from src.rerank import Reranker
        reranker = Reranker(model_name=_RERANKER_MODEL)
        print(f"reranker ready in {time.time() - t0:.1f}s", flush=True)

    failures = 0
    for expected_ticker, query in SAMPLE_QUERIES:
        print(f"\nQ: {query}")

        t0 = time.time()
        bm25 = hr.retrieve_bm25_only(query, k=20)
        t_bm25 = (time.time() - t0) * 1000
        print(f"  BM25({len(bm25)}) {t_bm25:5.0f}ms   [{', '.join(_tickers(bm25)[:6])}]")

        t0 = time.time()
        dense = hr.retrieve_dense_only(query, k=20)
        t_dense = (time.time() - t0) * 1000
        print(f"  DENSE({len(dense)}) {t_dense:4.0f}ms   [{', '.join(_tickers(dense)[:6])}]")

        t0 = time.time()
        hybrid = hr.retrieve(query, k=20)
        t_hybrid = (time.time() - t0) * 1000
        print(f"  HYBRID({len(hybrid)}) {t_hybrid:3.0f}ms   [{', '.join(_tickers(hybrid)[:6])}]")

        if reranker is not None:
            t0 = time.time()
            reranked = reranker.rerank(query, hybrid, top_n=5)
            t_rerank = (time.time() - t0) * 1000
            print(f"  RERANK({len(reranked)}) {t_rerank:3.0f}ms  "
                  f"[{', '.join(_tickers(reranked))}]")

        top = _tickers(hybrid)
        if expected_ticker and expected_ticker not in top:
            print(f"  WARN: {expected_ticker} missing from hybrid top-20")
            failures += 1

    AMBIGUOUS = "supply chain risks"
    print(f"\nQ (rewrite): {AMBIGUOUS}")
    from src.query_rewrite import QueryRewriter
    qr = QueryRewriter()
    t0 = time.time()
    rewritten = qr.rewrite(AMBIGUOUS)
    t_rewrite = (time.time() - t0) * 1000
    print(f"  REWRITE({len(rewritten)}) {t_rewrite:4.0f}ms  [{', '.join(rewritten[:3])}...]")
    t0 = time.time()
    expanded = hr.retrieve_multi([AMBIGUOUS] + rewritten, k=20)
    t_expand = (time.time() - t0) * 1000
    print(f"  EXPANDED({len(expanded)}) {t_expand:3.0f}ms  [{', '.join(_tickers(expanded)[:6])}]")

    print(f"\nRESULT: {'PASS' if failures == 0 else f'{failures} warnings'}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())