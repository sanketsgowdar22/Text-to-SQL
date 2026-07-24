# 12 — Security Documentation

Security policies, credential management, and vulnerability handling.

---

## Security Architecture

| Concern | Implementation |
|---------|---------------|
| **Authentication** | OAuth2 Bearer JWT (Access + Refresh tokens) |
| **Password Hashing** | bcrypt via passlib |
| **Credential Storage** | Fernet symmetric encryption (Phase 3) |
| **SQL Injection Prevention** | Read-only connection pools + AST validation (Phase 7) |
| **RBAC** | Role-based access control (Standard User, Admin/Superuser) |
| **CORS** | Configurable allowed origins via environment |
| **Request Tracing** | UUID `X-Request-ID` header injection |

---

## Security Principles

1. **Defense in Depth**: Multiple security layers (auth → RBAC → query validation → read-only execution)
2. **Least Privilege**: Users access only their own resources; superuser role required for admin operations
3. **Secrets Management**: All secrets loaded from environment variables, never hardcoded
4. **Credential Encryption**: Target database passwords encrypted at rest using Fernet
5. **Query Isolation**: Generated SQL validated via AST parsing before execution

---

## Related Documentation

- [System Architecture — Security Section](../../docs/architecture.md)
- [Phase 02 — Authentication](../05_Phases/PHASE_02_AUTHENTICATION.md)
- [Phase 07 — Guardrails](../05_Phases/PHASE_07_GUARDRAILS.md)
