# 09 — Database Documentation

Database schema designs, migration strategies, and data model documentation.

---

## Contents

| Document | Description |
|----------|-------------|
| [Database Schema & ERD](../../docs/database_schema.md) | System database entity-relationship diagram and table specifications |

---

## Database Strategy

| Aspect | Approach |
|--------|----------|
| **Development DB** | SQLite via `aiosqlite` |
| **Production DB** | PostgreSQL (planned) |
| **Migration Tool** | Alembic |
| **ORM** | SQLAlchemy 2.0 (Async) |
| **Naming Convention** | Enforced via `DB_NAMING_CONVENTION` in `app/core/constants.py` |

---

## Related Documentation

- [System Architecture](../../docs/architecture.md)
- [Phase 03 — Database Connections](../05_Phases/PHASE_03_DATABASE_CONNECTIONS.md)
- [Phase 04 — Schema Ingestion](../05_Phases/PHASE_04_SCHEMA_INGESTION.md)
