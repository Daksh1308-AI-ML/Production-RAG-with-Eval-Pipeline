"""Week 1 validation: basic dense retrieval against the live Qdrant corpus."""

import time

from src.config import config
from src.store import VectorStore

SAMPLE_QUERIES = [
    ("AAPL", "What were Apple's total net sales in fiscal 2024?"),
    ("MSFT", "How much revenue did Microsoft cloud generate?"),
    ("GOOGL", "What is Alphabet's total advertising revenue?"),
    ("AMZN", "What operating segments does Amazon report?"),
    ("NVDA", "What was NVIDIA's data center revenue?"),
    (None, "What supply chain and manufacturing risk factors do these companies disclose?"),
]


def main() -> int:
    vs = VectorStore()
    failures = 0

    for expected_ticker, query in SAMPLE_QUERIES:
        t0 = time.time()
        docs = vs.search(query, k=5)
        latency = (time.time() - t0) * 1000

        print(f"\nQ: {query}")
        print(f"   ({latency:.0f} ms)")
        tickers = []
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            tickers.append(m.get("ticker"))
            print(f"   {i}. [{m.get('ticker')} | {m.get('filing_date')} | "
                  f"chunk {m.get('chunk_index')} | score {m.get('score'):.4f}] "
                  f"{doc.page_content[:90]}...")

        if expected_ticker and expected_ticker not in tickers:
            print(f"   FAIL: expected top-5 to include {expected_ticker}")
            failures += 1

    t0 = time.time()
    filtered = vs.search("What were Apple's operating expenses?",
                         k=5, filter_metadata={"ticker": "AAPL"})
    filtered_ms = (time.time() - t0) * 1000
    bad = [d.metadata.get("ticker") for d in filtered if d.metadata.get("ticker") != "AAPL"]
    print(f"\nFilter test (ticker=AAPL, {filtered_ms:.0f} ms): {len(filtered)} hits")
    for d in filtered:
        print(f"   [{d.metadata.get('ticker')} | score {d.metadata.get('score'):.4f}] "
              f"{d.page_content[:80]}...")
    if bad:
        print(f"   FAIL: filtered results leaked non-AAPL tickers: {bad}")
        failures += 1

    print(f"\nRESULT: {'PASS' if failures == 0 else f'FAIL ({failures} failures)'}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())