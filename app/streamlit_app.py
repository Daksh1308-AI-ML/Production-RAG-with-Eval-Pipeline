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

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
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
