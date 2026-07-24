# Phase 03 — Multi-Database Connection Management

> **Status**: ⬜ Planned
> **Target Start**: 2026-07-28
> **Target End**: 2026-07-30
> **Dependencies**: Phase 02

---

## Overview

Phase 03 implements the database connection management system allowing users to securely store, test, and manage connections to external databases (PostgreSQL, MySQL, SQLite) that will be targeted by Text-to-SQL queries.

---

## Objectives

1. Create DatabaseConnection ORM model with encrypted credential storage
2. Implement Fernet-based credential vault for password encryption
3. Build dynamic SQLAlchemy engine factory for multiple database dialects
4. Create CRUD service layer for connection management
5. Implement connection testing and health check functionality
6. Build router with all connection management endpoints

---

## Tasks

| Task ID | Description | Files | Status |
|---------|-------------|-------|--------|
| 3.1 | DatabaseConnection ORM Model | `app/features/connections/models.py` | ⬜ |
| 3.2 | Fernet Credential Vault | `app/core/encryption.py` | ⬜ |
| 3.3 | Connection Engine Inspector | `app/features/connections/engine.py` | ⬜ |
| 3.4 | Connection Pydantic Schemas | `app/features/connections/schemas.py` | ⬜ |
| 3.5 | Connection Service Layer | `app/features/connections/service.py` | ⬜ |
| 3.6 | Connection Router Endpoints | `app/features/connections/router.py` | ⬜ |
| 3.7 | Connection Test Suite | `tests/unit/test_connection_*.py` | ⬜ |

---

## Acceptance Criteria

- [ ] Users can create, list, update, and delete database connections
- [ ] Passwords are encrypted at rest using Fernet
- [ ] Connection testing validates connectivity to target databases
- [ ] SQLite, PostgreSQL, and MySQL dialects supported
- [ ] All tests passing

---

## Definition of Done

- [ ] All tasks completed
- [ ] All tests passing
- [ ] API documentation updated
- [ ] PR reviewed and merged
