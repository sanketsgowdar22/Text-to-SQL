# Phase 04 — Schema Ingestion & Introspection Engine

> **Status**: ⬜ Planned
> **Target Start**: 2026-07-31
> **Target End**: 2026-08-02
> **Dependencies**: Phase 03

---

## Overview

Phase 04 implements the schema introspection engine that automatically extracts table structures, column types, primary keys, and foreign key relationships from connected databases. This metadata powers the RAG context for LLM prompt construction.

---

## Objectives

1. Create SchemaMetadata ORM model for persisting introspected schema data
2. Build multi-dialect introspector (SQLite, PostgreSQL, MySQL)
3. Implement async schema synchronization engine
4. Create LLM-optimized schema formatter for prompt injection
5. Build schema router endpoints

---

## Tasks

| Task ID | Description | Files | Status |
|---------|-------------|-------|--------|
| 4.1 | Schema Metadata ORM Model | `app/features/schema/models.py` | ⬜ |
| 4.2 | Multi-Dialect Introspector | `app/features/schema/introspector.py` | ⬜ |
| 4.3 | Async Schema Sync Engine | `app/features/schema/sync.py` | ⬜ |
| 4.4 | Schema LLM Prompt Formatter | `app/features/schema/formatter.py` | ⬜ |
| 4.5 | Schema Router Endpoints | `app/features/schema/router.py` | ⬜ |
| 4.6 | Schema Introspection Tests | `tests/unit/test_schema_*.py` | ⬜ |

---

## Acceptance Criteria

- [ ] Schema introspection extracts tables, columns, types, PKs, FKs
- [ ] Schema data persisted to `schema_metadata` table
- [ ] Schema formatter produces compact DDL-like text for LLM prompts
- [ ] Sync endpoint triggers background introspection
- [ ] All tests passing
