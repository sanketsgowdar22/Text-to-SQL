# 10 — Testing Documentation

Testing strategy, coverage requirements, and test architecture documentation.

---

## Testing Strategy

### Test Pyramid

| Level | Scope | Tool | Location |
|-------|-------|------|----------|
| **Unit** | Individual functions, classes, modules | pytest + pytest-asyncio | `backend/tests/unit/` |
| **Integration** | API endpoints, database operations | pytest + HTTPX AsyncClient | `backend/tests/unit/` (endpoint tests) |
| **E2E** | Full user workflows across services | *(Planned — Phase 8)* | `tests/` (top-level) |

### Coverage Requirements

- **Minimum Coverage**: 80% (enforced via `pyproject.toml`)
- **Coverage Tool**: pytest-cov
- **Coverage Report**: `pytest --cov=app --cov-report=html`

### Current Test Suite

| Test Module | Tests | Description |
|-------------|-------|-------------|
| `test_config.py` | Configuration validation | Settings loading, defaults, env overrides |
| `test_exceptions.py` | Exception handling | Custom exception hierarchy, error responses |
| `test_health.py` | Health endpoints | `/health` and `/health/ready` probes |
| `test_security.py` | Security utilities | Password hashing, JWT token lifecycle |
| `test_validators.py` | Input validation | Email, password, string sanitization |
| `test_user_model.py` | User ORM model | Persistence, defaults, query operations |

---

## Related Documentation

- [Engineering Rules](../03_Engineering/ENGINEERING_RULES.md)
- [Testing Report Template](../16_Templates/TESTING_REPORT_TEMPLATE.md)
