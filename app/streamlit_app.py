"""Streamlit UI for Production RAG."""

import streamlit as st

st.title("Production RAG - SEC 10-K Filings")
st.markdown("Ask questions about SEC 10-K filings.")

# Placeholder for chat interface
query = st.text_input("Enter your question:")

if query:
    st.info("RAG pipeline not yet initialized. Complete Week 1 setup first.")
