# Database Management

> **Status**: Active

---

## Overview

This directory contains database management scripts, seed data, and migration utilities for the ADIP platform.

---

## Current Setup

| Aspect | Configuration |
|--------|--------------|
| Development Database | SQLite (`backend/adip.db`) |
| Test Database | SQLite (`backend/test_adip.db`) |
| Production Database | PostgreSQL (planned) |
| Migration Tool | Alembic (`backend/alembic/`) |
| ORM | SQLAlchemy 2.0 (Async) |

---

## Migration Management

Migrations are managed via Alembic from the `backend/` directory:

```bash
cd backend

# Generate a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1

# View migration history
alembic history
```

---

## Related Documentation

- [Database Schema](../docs/database_schema.md)
- [Database Documentation](../docs/09_Database/README.md)
