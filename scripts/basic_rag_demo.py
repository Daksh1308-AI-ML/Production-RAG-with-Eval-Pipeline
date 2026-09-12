"""Week 1 demo: basic RAG chain (dense retrieve -> LLM answer with citations)."""

from src.config import config
from src.generate import AnswerGenerator
from src.store import VectorStore

DEMO_QUERIES = [
    "What were Apple's total net sales in fiscal 2024?",
    "How much revenue did Microsoft cloud generate?",
    "What was NVIDIA's data center revenue?",
]


def main() -> None:
    vs = VectorStore()
    generator = AnswerGenerator()

    for query in DEMO_QUERIES:
        docs = vs.search(query, k=5)
        print(f"\n=== Q: {query} ===")
        print(generator.generate(query, docs))
        print("\nSources:")
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            print(f"  [{i}] {m.get('ticker')} | {m.get('filing_date')} | "
                  f"chunk {m.get('chunk_index')} | "
                  f"{doc.page_content[:80]}...")


if __name__ == "__main__":
    main()