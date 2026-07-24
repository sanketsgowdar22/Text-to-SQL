# Frontend — ADIP Web Dashboard

> **Status**: ⬜ Planned (Phase 08)
> **Framework**: React / Next.js (to be decided at Phase 08 kickoff)

---

## Overview

The frontend provides a modern web dashboard for interacting with the AI Data Intelligence Platform. Users will be able to register, manage database connections, submit natural language queries, and view SQL results through a glassmorphism-styled interface.

---

## Planned Features

1. **Authentication UI** — Login, registration, session management
2. **Database Connection Manager** — Add, test, and manage target database connections
3. **NL-to-SQL Playground** — Natural language query input, generated SQL editor, tabular results viewer
4. **Dashboard** — Query history, usage statistics

---

## Technology Decision

The frontend framework will be selected at Phase 08 kickoff. Current candidates:

| Framework | Pros | Cons |
|-----------|------|------|
| Next.js (React) | SSR, file-based routing, large ecosystem | Heavier bundle |
| Vite + React | Fast dev server, minimal config | No SSR by default |

---

## Directory Structure (Planned)

```
frontend/
├── src/
│   ├── components/          # Shared UI components
│   ├── pages/               # Route pages
│   │   ├── auth/            # Login, Register
│   │   ├── connections/     # Connection management
│   │   └── playground/      # NL-to-SQL interface
│   ├── hooks/               # Custom React hooks
│   ├── services/            # API client services
│   ├── styles/              # Global styles, theme
│   └── utils/               # Utility functions
├── public/                  # Static assets
├── package.json
└── README.md
```

---

## Related Documentation

- [Phase 08 — Frontend & Deployment](../docs/05_Phases/PHASE_08_FRONTEND_DEPLOYMENT.md)
- [API Specification](../docs/api_specification.md)
- [System Architecture](../docs/architecture.md)
