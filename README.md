# financial-regulatory-RAG-assistant
FinPolicy Hybrid RAG: a Financial &amp; Regulatory Compliance Assistant

![Status](https://img.shields.io/badge/Status-Active%20Development-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)
![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-8E75B2)

A production-oriented **Retrieval-Augmented Generation (RAG)** system designed to query multi-page financial compliance and regulatory policy PDFs (such as Bank Negara Malaysia policy documents and fintech SOPs) with **hybrid retrieval, strict hallucination guardrails, and verifiable page-level citations**.

---
## System Architecture
*will be uploaded*

## Features
* **Hybrid Retrieval (Dense + Sparse):** Combines semantic similarity via **ChromaDB** with exact keyword matching via **BM25 (`rank_bm25`)** using **Reciprocal Rank Fusion (RRF)**—ensuring domain-specific acronyms (e.g., *e-KYC*, *AML/CFT*) are never missed.
* **Hallucination Guardrails:** Enforces a configurable cosine-similarity cutoff (`threshold = 0.30`) before invoking the LLM. Out-of-scope queries trigger an automated refusal instead of ungrounded generation.
* **Verifiable Source Citations:** Preserves document filenames and exact page numbers in vector metadata, instructing the LLM to append `[Filename, Page X]` inline citations to every factual claim.
* **Decoupled API & UI:** Exposes retrieval and generation through an async-ready **FastAPI** REST service (`Swagger UI` at `/docs`) paired with an interactive **Streamlit** frontend that displays expandable source chunks.
* **Automated Retrieval Benchmarking:** Includes an evaluation harness (`eval.py`) to measure **Top-K Retrieval Hit Rate** across domain-specific benchmark queries.

---

### Current Milestones
- [x] PDF ingestion pipeline with overlapping character chunking & page metadata
- [x] Hybrid retrieval engine (ChromaDB + BM25 with Reciprocal Rank Fusion)
- [x] FastAPI inference endpoint & Streamlit evidence viewer

  


