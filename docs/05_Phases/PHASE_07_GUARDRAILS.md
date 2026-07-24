# Phase 07 — SQL Execution Safety & Guardrails

> **Status**: ⬜ Planned
> **Target Start**: 2026-08-10
> **Target End**: 2026-08-12
> **Dependencies**: Phase 06

---

## Overview

Phase 07 implements the safety layer that validates and executes generated SQL queries. An AST-based validator ensures only read-only (SELECT) queries are executed, with enforcement of timeout caps and row limits.

---

## Objectives

1. Build AST-based read-only SQL validator using SQLGlot
2. Implement safe query executor with timeout and row limits
3. Create structured result formatter with execution metrics
4. Build query execution endpoint

---

## Tasks

| Task ID | Description | Files | Status |
|---------|-------------|-------|--------|
| 7.1 | AST Read-Only Validator | `app/features/execution/validator.py` | ⬜ |
| 7.2 | Safe Query Executor | `app/features/execution/executor.py` | ⬜ |
| 7.3 | Result Formatter & Metrics | `app/features/execution/formatter.py` | ⬜ |
| 7.4 | Query Execution Router | `app/features/execution/router.py` | ⬜ |
| 7.5 | Execution Safety Tests | `tests/unit/test_execution_*.py` | ⬜ |

---

## Acceptance Criteria

- [ ] SELECT queries pass validation
- [ ] DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE queries rejected
- [ ] Multi-statement injection attempts blocked
- [ ] Query timeout enforced (configurable, default 30s)
- [ ] Row limit enforced (configurable, default 1000)
- [ ] Results formatted with column headers, rows, count, latency
- [ ] All tests passing
