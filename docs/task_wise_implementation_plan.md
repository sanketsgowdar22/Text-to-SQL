# Task-Wise Implementation Plan for Project Phases

This document provides a granular, task-by-task execution specification for every phase of the **AI Data Intelligence Platform (ADIP)** development lifecycle. Each phase is broken down into discrete sub-tasks, targeted files, key requirements, and verification criteria.

---

## Index of Phases

- [Phase 0: System Architecture & Documentation Suite](#phase-0-system-architecture--documentation-suite-)
- [Phase 1: Core Foundation & Shared Services](#phase-1-core-foundation--shared-services-)
- [Phase 2: Authentication & User Management](#phase-2-authentication--user-management-target-)
- [Phase 3: Multi-Database Connection Management](#phase-3-multi-database-connection-management)
- [Phase 4: Schema Ingestion & Introspection Engine](#phase-4-schema-ingestion--introspection-engine)
- [Phase 5: Embeddings & Vector Store RAG Retrieval](#phase-5-embeddings--vector-store-rag-retrieval)
- [Phase 6: Gemini LLM Text-to-SQL Engine](#phase-6-gemini-llm-text-to-sql-engine)
- [Phase 7: SQL Execution Safety & Guardrails](#phase-7-sql-execution-safety--guardrails)
- [Phase 8: Frontend Dashboard & Production Deployment](#phase-8-frontend-dashboard--production-deployment)

---

## Phase 0: System Architecture & Documentation Suite ✅

| Task ID | Task Description | Primary Files | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **0.1** | System Topology & Architecture Design | `docs/architecture.md` | Mermaid topology diagram, layer breakdown, tech stack summary |
| **0.2** | PRD & Milestone Roadmap | `docs/prd_and_roadmap.md` | Product requirements, Gantt timeline, phase deliverables summary |
| **0.3** | Internal Relational Data Model | `docs/database_schema.md` | Mermaid ERD, column constraints, data types, indexes |
| **0.4** | REST API Endpoint Specification | `docs/api_specification.md` | Request/response JSON schemas, error codes, HTTP status codes |
| **0.5** | Task-Wise Implementation Guide | `docs/task_wise_implementation_plan.md` | Detailed task-by-task breakdown for all 8 phases |
| **0.6** | Root Documentation Portal | `README.md` | Links to all `docs/*` blueprints and development quickstart |

---

## Phase 1: Core Foundation & Shared Services ✅

| Task ID | Task Description | Primary Files | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **1.1** | Project Directory & Dependency Setup | `pyproject.toml`, `requirements.txt`, `requirements-dev.txt` | Virtual environment initialized with FastAPI, SQLAlchemy 2.0, Pydantic v2 |
| **1.2** | Environment Configuration | `app/core/config.py`, `.env.example` | Pydantic `BaseSettings` loading `.env` variables with defaults |
| **1.3** | Async Database Engine Setup | `app/core/database.py` | SQLAlchemy `create_async_engine`, `async_sessionmaker`, `init_db`, `close_db` |
| **1.4** | Base ORM Models & Mixins | `app/shared/models.py` | `UUIDMixin` (UUIDv4 primary keys), `TimestampMixin` (`created_at`, `updated_at`) |
| **1.5** | Standardized Response Schemas | `app/shared/schemas.py`, `app/shared/utils.py` | Generic `SuccessResponse[T]`, `ErrorResponse`, `HealthResponse`, logger setup |
| **1.6** | Centralized Exception Infrastructure | `app/core/exceptions.py` | Custom exception hierarchy (`AppException`, `NotFoundException`, etc.) & FastAPI handlers |
| **1.7** | Application Middleware & Factory | `app/core/middleware.py`, `app/main.py` | Request ID injection, CORS setup, application factory `create_app()` |
| **1.8** | Health & Readiness Probes | `app/main.py`, `tests/unit/test_health.py` | GET `/api/v1/health` & `/api/v1/health/ready` endpoints returning status |
| **1.9** | Baseline Test Suite | `tests/conftest.py`, `tests/unit/test_*.py` | 63 unit tests passing with full baseline coverage |

---

## Phase 2: Authentication & User Management (Target 🎯)

| Task ID | Task Description | Primary Files | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **2.1** | User ORM Model | `app/features/auth/models.py` | `User` model with `email`, `hashed_password`, `full_name`, `is_active`, `is_superuser` |
| **2.2** | Alembic DB Migration | `alembic/versions/001_create_users_table.py` | Migration script creating `users` table with unique index on `email` |
| **2.3** | Auth Request/Response Schemas | `app/features/auth/schemas.py` | `UserRegisterRequest`, `UserLoginRequest`, `TokenResponse`, `UserResponse`, `UserUpdateRequest` |
| **2.4** | Password Hashing & JWT Utils | `app/core/security.py` | `hash_password()`, `verify_password()`, `create_access_token()`, `create_refresh_token()` |
| **2.5** | Auth Service Layer | `app/features/auth/service.py` | `register_user()`, `authenticate_user()`, `create_user_tokens()`, `refresh_user_token()` |
| **2.6** | Security Dependency Guards | `app/features/auth/dependencies.py` | `get_current_user()` (decodes JWT & loads DB user), `get_current_superuser()` |
| **2.7** | Auth Router Endpoints | `app/features/auth/router.py` | Endpoints: POST `/register`, POST `/login`, POST `/refresh`, GET `/me`, PUT `/me` |
| **2.8** | Router Wire-up | `app/main.py` | Register `auth_router` in `_register_routers()` under `/api/v1/auth` |
| **2.9** | Auth Test Suite | `tests/unit/test_auth_service.py`, `tests/unit/test_auth_router.py` | Complete unit & endpoint integration test coverage |

---

## Phase 3: Multi-Database Connection Management

| Task ID | Task Description | Primary Files | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **3.1** | Database Connection ORM Model | `app/features/connections/models.py` | `DatabaseConnection` model storing dialect, host, port, db_name, encrypted password |
| **3.2** | Fernet Credential Vault | `app/core/encryption.py` | `encrypt_secret()`, `decrypt_secret()` using Fernet key |
| **3.3** | Connection Engine Inspector | `app/features/connections/engine.py` | Dynamic SQLAlchemy engine factory for SQLite, PostgreSQL, MySQL |
| **3.4** | Connection Pydantic Schemas | `app/features/connections/schemas.py` | `ConnectionCreate`, `ConnectionResponse`, `ConnectionUpdate`, `TestConnectionResponse` |
| **3.5** | Connection Service Layer | `app/features/connections/service.py` | Create, list, retrieve, update, delete, and test connectivity |
| **3.6** | Connection Router Endpoints | `app/features/connections/router.py` | GET `/connections`, POST `/connections`, GET `/{id}`, PUT `/{id}`, DELETE `/{id}`, POST `/{id}/test` |
| **3.7** | Connection Test Suite | `tests/unit/test_connection_service.py`, `tests/unit/test_connection_router.py` | Test CRUD operations and mock database ping validation |

---

## Phase 4: Schema Ingestion & Introspection Engine

| Task ID | Task Description | Primary Files | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **4.1** | Schema Metadata ORM Model | `app/features/schema/models.py` | `SchemaMetadata` storing table name, columns JSON, PK JSON, FK JSON |
| **4.2** | Multidialect Introspector | `app/features/schema/introspector.py` | Extract table names, column data types, nullable flags, primary/foreign keys |
| **4.3** | Async Schema Sync Engine | `app/features/schema/sync.py` | Asynchronously inspect target database and persist structured JSON to `schema_metadata` |
| **4.4** | Schema LLM Prompt Formatter | `app/features/schema/formatter.py` | Format schema metadata into compact DDL-like text for LLM prompts |
| **4.5** | Schema Router Endpoints | `app/features/schema/router.py` | POST `/schema/{connection_id}/sync` & GET `/schema/{connection_id}` |
| **4.6** | Schema Introspection Tests | `tests/unit/test_schema_introspector.py` | Verify schema extraction against SQLite sample databases |

---

## Phase 5: Embeddings & Vector Store RAG Retrieval

| Task ID | Task Description | Primary Files | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **5.1** | Embedding Generator | `app/features/embeddings/encoder.py` | Sentence-Transformers (`all-MiniLM-L6-v2`) text embedder |
| **5.2** | FAISS Vector Store Manager | `app/features/embeddings/vector_store.py` | FAISS index creation, vector insertion, persistence, loading |
| **5.3** | Semantic Retriever | `app/features/embeddings/retriever.py` | Search top-K relevant tables/columns for a natural language prompt |
| **5.4** | RAG Context Builder | `app/features/embeddings/context.py` | Filter and assemble relevant schema subset into LLM context window |
| **5.5** | Vector Index Test Suite | `tests/unit/test_embeddings.py` | Verify vector encoding, similarity search scoring, and index save/load |

---

## Phase 6: Gemini LLM Text-to-SQL Engine

| Task ID | Task Description | Primary Files | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **6.1** | Prompt Engineering Module | `app/features/query/prompts.py` | System prompt, dialect rules, RAG schema injection, few-shot examples |
| **6.2** | Google Gemini API Client | `app/features/query/llm_client.py` | `GoogleGenerativeAI` wrapper targeting `gemini-1.5-flash` with retry logic |
| **6.3** | SQL Output Parser | `app/features/query/parser.py` | Extract clean SQL statement from Markdown code blocks & separate explanation |
| **6.4** | Query History ORM Model | `app/features/query/models.py` | `QueryLog` storing prompt, generated SQL, validation status, execution timing |
| **6.5** | Text-to-SQL Router | `app/features/query/router.py` | POST `/query/generate` endpoint returning generated SQL and explanation |
| **6.6** | Text-to-SQL Test Suite | `tests/unit/test_query_generator.py` | Test prompt formatting, mock LLM responses, SQL extraction |

---

## Phase 7: SQL Execution Safety & Guardrails

| Task ID | Task Description | Primary Files | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **7.1** | AST Read-Only Validator | `app/features/execution/validator.py` | SQLGlot parser verifying queries contain only `SELECT` statement AST nodes |
| **7.2** | Safe Query Executor | `app/features/execution/executor.py` | Execute query against target DB with statement timeout cap (e.g. 5s) and row limit (e.g. 1000) |
| **7.3** | Result Formatter & Metrics | `app/features/execution/formatter.py` | Format tabular results (column headers, rows, row count, execution latency) |
| **7.4** | Query Execution Router | `app/features/execution/router.py` | POST `/query/execute` endpoint |
| **7.5** | Execution Safety Tests | `tests/unit/test_execution_safety.py` | Test rejection of `DROP`, `DELETE`, `UPDATE`, `INSERT`, multi-statements |

---

## Phase 8: Frontend Dashboard & Production Deployment

| Task ID | Task Description | Primary Files | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **8.1** | Web Client Boilerplate | `frontend/` (React / Next.js) | Modern glassmorphism UI layout, dark mode, Google Fonts |
| **8.2** | Auth & Connection UI | `frontend/src/pages/auth`, `frontend/src/pages/connections` | Login/Register forms, database connection manager |
| **8.3** | NL-to-SQL Playground | `frontend/src/pages/playground` | Query input, SQL editor with syntax highlighting, tabular data viewer |
| **8.4** | Production Containerization | `Dockerfile`, `docker-compose.yml` | Multi-stage Docker build for backend and frontend |
| **8.5** | CI/CD Pipeline & Health Check | `.github/workflows/ci.yml` | Automated test runner and production deployment scripts |
