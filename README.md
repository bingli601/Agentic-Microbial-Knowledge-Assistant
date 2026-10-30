# Agentic Microbial Knowledge Assistant

A local microbial knowledge Q&A system based on PubMed literature using Retrieval-Augmented Generation (RAG).

This project develops a **fully local, zero-API-cost Retrieval-Augmented Generation (RAG) system** focused on microbial survival, predictive microbiology, and food safety. It implements a complete end-to-end workflow covering data collection, preprocessing, vectorization, retrieval, question answering, and basic evaluation. The project serves as a practical engineering portfolio demonstrating hands-on experience in AI/LLM application development.

---

## Project Highlights

- **Domain-Specific Knowledge:** Builds a knowledge base from approximately 907 PubMed abstracts related to microbiology and food safety.
- **Fully Local Deployment:** Embedding models, vector database, and large language models run locally without requiring paid APIs.
- **End-to-End Engineering:** Covers the complete pipeline from data collection → preprocessing → indexing → retrieval → generation → evaluation → Web demo.
- **Interactive Demonstration:** Provides a Streamlit-based interface for interactive demonstrations.

---

## System Architecture

```text
User Query
    ↓
Embedding (bge-small-en-v1.5)
    ↓
Vector Retrieval (Local Qdrant)
+ Optional Hybrid Search / Query Rewriting
    ↓
Retrieved Literature Passages (Context)
    ↓
Local LLM (Qwen2.5:7B via Ollama)
    ↓
Chinese Answer + Literature Sources (PMIDs)
```