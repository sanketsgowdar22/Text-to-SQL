# Tests — Cross-Service Integration & E2E Tests

> **Status**: ⬜ Planned (Phase 08)

---

## Overview

This directory will contain integration and end-to-end tests that span multiple services. Unit tests for individual services remain in their respective directories (e.g., `backend/tests/`).

---

## Test Hierarchy

| Level | Location | Scope |
|-------|----------|-------|
| **Unit** | `backend/tests/unit/` | Individual functions, classes |
| **Integration** | `backend/tests/` | API endpoints, database operations |
| **E2E** | `tests/` (this directory) | Full user workflows across services |

---

## Related Documentation

- [Testing Strategy](../docs/10_Testing/README.md)
- [Testing Report Template](../docs/16_Templates/TESTING_REPORT_TEMPLATE.md)
