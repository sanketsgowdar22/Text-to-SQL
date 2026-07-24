# AI/ML Service Layer

> **Status**: ⬜ Planned (Phases 05–06)

---

## Overview

The AI service layer houses the machine learning components of the ADIP platform including embedding generation, vector indexing, and LLM integration.

---

## Components (Planned)

| Component | Phase | Description |
|-----------|-------|-------------|
| Embedding Encoder | Phase 05 | Sentence-Transformers text-to-vector encoding |
| Vector Store | Phase 05 | FAISS index management and persistence |
| Semantic Retriever | Phase 05 | Top-K schema element retrieval |
| Prompt Engine | Phase 06 | LLM prompt construction pipeline |
| Gemini Client | Phase 06 | Google Gemini API integration |
| SQL Parser | Phase 06 | LLM response → clean SQL extraction |

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| LLM Provider | Google Gemini API (`gemini-1.5-flash`) |

> **Note**: AI components are currently implemented as feature modules within `backend/app/features/`. This top-level directory will house standalone AI service code if the architecture evolves to a microservices model.

---

## Related Documentation

- [Phase 05 — Embeddings](../docs/05_Phases/PHASE_05_EMBEDDINGS.md)
- [Phase 06 — Text-to-SQL](../docs/05_Phases/PHASE_06_TEXT_TO_SQL.md)
- [AI Documentation](../docs/08_AI/README.md)
