# Changelog

All notable changes to the ADIP project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Phase 00: Enterprise Engineering Foundation
  - Repository restructured with 19-category documentation hierarchy
  - Engineering Rules (`ENGINEERING_RULES.md`) — mandatory governance document
  - Task Implementation Guide (`TASK_IMPLEMENTATION_GUIDE.md`) — step-by-step workflow
  - Git Workflow (`GIT_WORKFLOW.md`) — branch model, commit conventions, merge strategy
  - Pull Request template with 15 structured sections
  - 5 Issue templates (Feature, Bug, Refactor, Documentation, Architecture)
  - 7 Document templates (Phase, Task, API Spec, ADR, Testing, Review, Release)
  - 9 Phase documents (Phase 00–08)
  - CI/CD pipeline (GitHub Actions: lint, type-check, test, build)
  - Docker configuration (Dockerfile.backend, docker-compose.yml)
  - Backend feature skeletons (connections, schema, embeddings, query, execution)
  - Service skeletons (frontend, AI, database, shared, scripts, tests)
  - Repository audit report and current problems analysis
  - Governance files (CHANGELOG, CONTRIBUTING, CODE_OF_CONDUCT, LICENSE)

---

## [0.1.0] — 2026-07-24

### Added
- Phase 0: System Architecture & Technical Documentation Suite
  - System architecture document with Mermaid topology diagram
  - Product Requirements Document & multi-phase roadmap
  - Database schema design with ERD
  - REST API specification with endpoint contracts
  - Task-wise implementation plan for all 8 phases
  - Root README with project structure and quickstart

- Phase 1: Core Foundation & Shared Services
  - FastAPI application factory with lifespan management
  - Pydantic Settings configuration from environment variables
  - Async SQLAlchemy 2.0 engine with session factory
  - Base ORM models (UUIDMixin, TimestampMixin)
  - Centralized exception hierarchy with FastAPI handlers
  - CORS, request logging, and request ID middleware
  - Health check and readiness probe endpoints
  - Standardized response schemas (SuccessResponse, ErrorResponse, PaginatedResponse)
  - Input validators (email, password, database name)
  - Pagination helpers
  - 63+ passing unit tests

- Phase 2 (partial): Authentication
  - User ORM model with email, hashed_password, full_name, is_active, is_superuser
  - Password hashing (bcrypt) and JWT token utilities
