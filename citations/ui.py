"""
Streamlit demo frontend
"""


import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="FinPolicy RAG Assistant", layout="centered")
st.title("Financial Policy & Compliance RAG")
st.caption("Hybrid Retrieval (ChromaDB Vector + BM25) with Citation Guardrails")

query = st.text_input("Ask a question about your uploaded policy PDFs:")
top_k = st.sidebar.slider("Retrieved Chunks (Top-K)", min_value=2, max_value=8, value=4)

if st.button("Ask") and query.strip():
    with st.spinner("Retrieving policy sections and verifying citations..."):
        resp = requests.post(API_URL, json={"query": query, "top_k": top_k}, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            st.markdown("### Answer")
            st.write(data["answer"])
            st.caption(f"Top Semantic Similarity Score: `{data['best_similarity']}`")

            if data["sources"]:
                st.markdown("### Retrieved Evidence")
                for src in data["sources"]:
                    with st.expander(f"{src['source']} — Page {src['page']}"):
                        st.write(src["text"])
        else:
            st.error(f"API Error: {resp.text}")