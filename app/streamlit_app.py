"""Streamlit UI for Production RAG."""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.chunker import DocumentChunker
from src.rag import RAGPipeline

STRATEGY_LABELS = {
    "full": "Full — rewrite + hybrid search + rerank (slowest, best quality)",
    "rerank": "Rerank — hybrid search + rerank (no query rewrite)",
    "hybrid": "Hybrid — dense + BM25 retrieval (no rewrite, no rerank)",
    "baseline": "Baseline — dense retrieval only (fastest)",
}


def analytics_view():
    """Analytics dashboard: storage, cache, latency, eval/AB results."""
    st.title("Analytics Dashboard")

    try:
        from src.store import VectorStore
        vs = VectorStore()
        points = vs.client.count(vs.collection_name).count
        cache_points = 0
        try:
            cache_points = vs.client.count(config.cache.collection_name).count
        except Exception:
            pass
        col1, col2, col3 = st.columns(3)
        col1.metric("Qdrant chunks (sec_filings)", f"{points:,}")
        col2.metric("Semantic cache entries", f"{cache_points:,}")
    except Exception as e:
        st.warning(f"Qdrant unavailable: {e}")
        points = 0
        cache_points = 0

    hit_rate = None
    if st.session_state.get("cache_stats"):
        stats = st.session_state.cache_stats
        total = stats["hits"] + stats["misses"]
        hit_rate = stats["hits"] / total if total else None

    st.subheader("Semantic cache")
    if hit_rate is not None:
        st.progress(hit_rate, text=f"Hit rate: {hit_rate:.0%} "
                                   f"({st.session_state.cache_stats['hits']} hits / "
                                   f"{st.session_state.cache_stats['misses']} misses — this session)")
    else:
        st.caption("No queries yet this session — hit rate appears after a cache lookup.")

    st.subheader("Session latency (ms)")
    latencies = [
        m["latency_ms"] for m in st.session_state.get("messages", [])
        if m.get("latency_ms") is not None
    ]
    if latencies:
        avg = sum(latencies) / len(latencies)
        st.metric("Average latency", f"{avg:.0f} ms", f"{len(latencies)} queries")
        st.bar_chart({"latency_ms": latencies})
    else:
        st.caption("No query latency recorded yet.")

    st.subheader("A/B evaluation results")
    ab_dir = Path("data/evaluation/results_ab")
    report = ab_dir / "ab_report.md"
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
    elif (ab_dir / "summary.json").exists():
        summary = json.load(open(ab_dir / "summary.json", encoding="utf-8"))
        st.json(summary)
    else:
        st.caption(
            "No results yet — run `python scripts/run_ab.py` then "
            "`python scripts/ab_report.py`."
        )


def chat_view():
    """Main chat UI."""
    # --- Pipeline cache (one per strategy) ---
    @st.cache_resource(show_spinner="Loading RAG pipeline...")
    def load_pipeline(_strategy: str):
        filings = json.load(open(config.processed_dir / "filings.json", encoding="utf-8"))
        chunks = DocumentChunker().chunk_documents(filings)
        pipeline = RAGPipeline(strategy=_strategy)
        pipeline.initialize(chunks)
        return pipeline

    try:
        pipeline = load_pipeline(strategy)
    except Exception as e:
        st.error(f"Failed to start RAG pipeline — is Qdrant / Ollama running?\n\n`{e}`")
        st.stop()

    # --- Chat state ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Production RAG - SEC 10-K Filings")
    st.caption("Ask questions about Apple, Microsoft, Google, Amazon, and NVIDIA 10-K filings.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg.get("latency_ms") is not None:
                st.caption(f"Latency: {msg['latency_ms']:.0f} ms")
            for src in msg.get("sources", []):
                with st.expander(f"Source: {src['source']} — {src['section']}"):
                    content = src["content"]
                    if len(content) > 600:
                        st.text(content[:600] + "… [truncated]")
                    else:
                        st.text(content)

    if user_input := st.chat_input("Ask a question about SEC 10-K filings..."):
        st.session_state.messages.append({"role": "user", "text": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    response = pipeline.query(user_input)
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.session_state.messages.append({"role": "assistant", "text": f"Error: {e}"})
                    st.stop()

            st.markdown(response.answer)
            st.caption(f"Latency: {response.latency_ms:.0f} ms")

            sources = response.sources[:5]
            for src in sources:
                with st.expander(f"Source: {src['source']} — {src['section']}"):
                    content = src["content"]
                    if len(content) > 600:
                        st.text(content[:600] + "… [truncated]")
                    else:
                        st.text(content)

        st.session_state.messages.append({
            "role": "assistant",
            "text": response.answer,
            "latency_ms": response.latency_ms,
            "sources": sources,
        })
        # Record cache hit/miss for analytics dashboard.
        try:
            cache = pipeline._get_cache()
            st.session_state.cache_stats = {"hits": cache.hits, "misses": cache.misses}
        except Exception:
            pass


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    mode = st.radio("View", ["Chat", "Analytics"], index=0)

strategy = "full"
if mode == "Chat":
    with st.sidebar:
        st.divider()
        strategy = st.selectbox(
            "Retrieval strategy",
            options=list(STRATEGY_LABELS.keys()),
            format_func=lambda s: STRATEGY_LABELS[s],
            index=0,
        )
        st.divider()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption("Backend: local Qdrant + Ollama")
    chat_view()
else:
    with st.sidebar:
        st.divider()
        st.caption("Backend: local Qdrant + Ollama")
    analytics_view()
