# 15 — Coding Standards

Coding conventions, style guidelines, and quality standards.

---

## Python Standards

### Enforced via Tooling

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **Ruff** | Linting & formatting | `pyproject.toml` — 12 rule categories enabled |
| **MyPy** | Static type checking | `pyproject.toml` — strict mode enabled |
| **pytest** | Test runner | `pyproject.toml` — asyncio_mode=auto, strict markers |

### General Conventions

1. **Line Length**: 100 characters maximum
2. **Quote Style**: Double quotes
3. **Indent Style**: 4 spaces
4. **Imports**: Sorted by isort rules (stdlib → third-party → first-party)
5. **Type Annotations**: Required on all function signatures (enforced by MyPy strict)
6. **Docstrings**: Required on all public classes and functions (Google style)

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Modules | `snake_case` | `config.py`, `database.py` |
| Classes | `PascalCase` | `ADIPBaseException`, `User` |
| Functions | `snake_case` | `get_settings()`, `hash_password()` |
| Constants | `SCREAMING_SNAKE_CASE` | `API_V1_PREFIX`, `MAX_PAGE_SIZE` |
| Private | `_leading_underscore` | `_build_error_response()` |
| Test Functions | `test_<description>` | `test_user_persistence_and_defaults` |

### Error Handling

- All custom exceptions inherit from `ADIPBaseException`
- Use specific exception types (`NotFoundException`, `ConflictException`, etc.)
- Never catch bare `Exception` in feature code
- All error responses use the `ErrorResponse` schema

### API Design

- All endpoints return `SuccessResponse[T]` or `ErrorResponse`
- Use `API_V1_PREFIX` for all route paths
- Use FastAPI dependency injection for database sessions and auth
- Paginated endpoints use `PaginationParams` dependency

---

## Documentation Standards

- All documents use Markdown format
- Use Mermaid for diagrams
- Use GitHub-style alerts (`[!NOTE]`, `[!IMPORTANT]`, `[!WARNING]`)
- Documents must have a title, date, and version where applicable
- Use tables for structured data instead of prose

---

## Related Documentation

- [Engineering Rules](../03_Engineering/ENGINEERING_RULES.md)
- [Review Process](../17_Reviews/README.md)
