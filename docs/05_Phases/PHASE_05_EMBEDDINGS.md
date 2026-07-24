# Phase 05 — Embeddings & Vector Store RAG Retrieval

> **Status**: ⬜ Planned
> **Target Start**: 2026-08-03
> **Target End**: 2026-08-05
> **Dependencies**: Phase 04

---

## Overview

Phase 05 implements the semantic search layer using sentence-transformers for embedding generation and FAISS for vector indexing. This enables the RAG pipeline to retrieve relevant schema context for natural language queries.

---

## Objectives

1. Build embedding generator using Sentence-Transformers (`all-MiniLM-L6-v2`)
2. Implement FAISS vector store with persistence (save/load)
3. Create semantic retriever for top-K relevant schema elements
4. Build RAG context assembler for LLM prompt construction

---

## Tasks

| Task ID | Description | Files | Status |
|---------|-------------|-------|--------|
| 5.1 | Embedding Generator | `app/features/embeddings/encoder.py` | ⬜ |
| 5.2 | FAISS Vector Store Manager | `app/features/embeddings/vector_store.py` | ⬜ |
| 5.3 | Semantic Retriever | `app/features/embeddings/retriever.py` | ⬜ |
| 5.4 | RAG Context Builder | `app/features/embeddings/context.py` | ⬜ |
| 5.5 | Vector Index Test Suite | `tests/unit/test_embeddings.py` | ⬜ |

---

## Acceptance Criteria

- [ ] Schema elements encoded as dense vectors
- [ ] FAISS index supports insert, search, save, and load
- [ ] Semantic retriever returns top-K relevant schemas for a query
- [ ] RAG context builder assembles schema subset for LLM prompt
- [ ] All tests passing
