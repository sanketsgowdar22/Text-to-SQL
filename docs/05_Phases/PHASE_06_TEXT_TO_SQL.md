# Phase 06 — NVIDIA NIM LLM Text-to-SQL Engine

> **Status**: ⬜ Planned
> **Target Start**: 2026-08-06
> **Target End**: 2026-08-09
> **Dependencies**: Phase 05

---

## Overview

Phase 06 implements the core Text-to-SQL generation engine using NVIDIA NIM API (z-ai/glm-5.3-flash model). The engine constructs prompts incorporating user queries, RAG-retrieved schema context, dialect-specific constraints, and few-shot examples to produce accurate SQL queries.

---

## Objectives

1. Design prompt engineering pipeline with system context, schema injection, and examples
2. Build NVIDIA NIM API client with retry logic
3. Implement SQL output parser for extracting clean SQL from LLM responses
4. Create QueryLog ORM model for audit logging
5. Build query generation endpoint

---

## Tasks

| Task ID | Description | Files | Status |
|---------|-------------|-------|--------|
| 6.1 | Prompt Engineering Module | `app/features/query/prompts.py` | ⬜ |
| 6.2 | NVIDIA NIM API Client | `app/features/query/llm_client.py` | ⬜ |
| 6.3 | SQL Output Parser | `app/features/query/parser.py` | ⬜ |
| 6.4 | Query History ORM Model | `app/features/query/models.py` | ⬜ |
| 6.5 | Text-to-SQL Router | `app/features/query/router.py` | ⬜ |
| 6.6 | Text-to-SQL Test Suite | `tests/unit/test_query_*.py` | ⬜ |

---

## Acceptance Criteria

- [ ] Prompts incorporate user query + RAG schema context + dialect rules
- [ ] NVIDIA NIM API integration with retry and error handling
- [ ] SQL extracted from LLM markdown code blocks
- [ ] Query logs persisted with prompt, SQL, validation status
- [ ] All tests passing
