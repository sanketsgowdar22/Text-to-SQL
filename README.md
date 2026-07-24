# AI Data Intelligence Platform (ADIP) — Text-to-SQL

Enterprise AI Data Intelligence Platform powered by FastAPI, Google Gemini LLM, Sentence-Transformers, FAISS, and SQLAlchemy.

[![CI Pipeline](https://github.com/your-org/Text-to-SQL/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/Text-to-SQL/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

## 📚 Documentation Suite

### Engineering Governance

| Document | Description |
| :--- | :--- |
| **[Engineering Rules](docs/03_Engineering/ENGINEERING_RULES.md)** | Mandatory rules for all implementation work |
| **[Task Implementation Guide](docs/03_Engineering/TASK_IMPLEMENTATION_GUIDE.md)** | Step-by-step workflow for every task |
| **[Git Workflow](docs/13_Git/GIT_WORKFLOW.md)** | Branch model, commit conventions, merge strategy |
| **[Contributing Guide](CONTRIBUTING.md)** | How to contribute to this project |

### Technical Documentation

| Document | Description |
| :--- | :--- |
| **[System Architecture](docs/architecture.md)** | Multi-tier topology, component breakdown, security model |
| **[PRD & Roadmap](docs/prd_and_roadmap.md)** | Product vision, requirements, and 8-phase roadmap |
| **[Database Schema](docs/database_schema.md)** | System database ERD and data models |
| **[API Specification](docs/api_specification.md)** | RESTful API endpoint contracts and schemas |
| **[Repository Structure](docs/04_Architecture/REPOSITORY_STRUCTURE.md)** | Complete directory tree specification |

### Phase Documents

| Phase | Document | Status |
| :--- | :--- | :--- |
| Phase 00 | [Engineering Foundation](docs/05_Phases/PHASE_00_ENGINEERING_FOUNDATION.md) | ✅ Complete |
| Phase 01 | [Project Foundation](docs/05_Phases/PHASE_01_PROJECT_FOUNDATION.md) | ✅ Complete |
| Phase 02 | [Authentication](docs/05_Phases/PHASE_02_AUTHENTICATION.md) | 🎯 In Progress |
| Phase 03 | [Database Connections](docs/05_Phases/PHASE_03_DATABASE_CONNECTIONS.md) | ⬜ Planned |
| Phase 04 | [Schema Ingestion](docs/05_Phases/PHASE_04_SCHEMA_INGESTION.md) | ⬜ Planned |
| Phase 05 | [Embeddings & RAG](docs/05_Phases/PHASE_05_EMBEDDINGS.md) | ⬜ Planned |
| Phase 06 | [Text-to-SQL Engine](docs/05_Phases/PHASE_06_TEXT_TO_SQL.md) | ⬜ Planned |
| Phase 07 | [Safety & Guardrails](docs/05_Phases/PHASE_07_GUARDRAILS.md) | ⬜ Planned |
| Phase 08 | [Frontend & Deployment](docs/05_Phases/PHASE_08_FRONTEND_DEPLOYMENT.md) | ⬜ Planned |

### Full Documentation Index

Browse the complete [19-category documentation hierarchy](docs/) covering Project, Product, Engineering, Architecture, Phases, Tasks, API, AI, Database, Testing, Deployment, Security, Git, Development, Standards, Templates, Reviews, Releases, and Decisions.

---

## 🚀 Quick Start

### Backend Setup

```bash
cd backend

# Create & activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configure environment
copy .env.example .env

# Start development server
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Setup

```bash
docker compose -f docker/docker-compose.yml up --build
```

### Running Tests

```bash
cd backend
.\venv\Scripts\python.exe -m pytest -v
```

### API Documentation

Once running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🛠️ Project Structure

```
Text-to-SQL/
├── .github/                  # CI/CD, PR & Issue Templates
├── docs/                     # 19-Category Documentation Suite
│   ├── audit/                # Repository Audit Reports
│   ├── 01_Project/           # Project Governance
│   ├── 03_Engineering/       # Engineering Rules & Guides
│   ├── 05_Phases/            # Phase Specification Documents
│   ├── 13_Git/               # Git Workflow
│   ├── 16_Templates/         # Reusable Document Templates
│   └── ...                   # 13 more categories
├── backend/                  # FastAPI Backend Service
│   ├── app/
│   │   ├── core/             # Config, Security, DB, Middleware
│   │   ├── features/         # Modular Feature Packages
│   │   └── shared/           # Common Models & Utilities
│   └── tests/                # Unit & Integration Tests
├── frontend/                 # Web Dashboard (Phase 08)
├── ai/                       # AI/ML Service Layer
├── docker/                   # Container Configuration
├── CHANGELOG.md              # Version History
├── CONTRIBUTING.md           # Contribution Guidelines
└── LICENSE                   # Apache 2.0
```

---

## 📋 Development Workflow

Every task follows this workflow — no exceptions:

```
Read Engineering Rules → Read Phase Doc → Read Task Doc →
Create Branch → Implement → Test → Update Docs →
Commit → Push → Create PR → Review → Merge → Delete Branch
```

See [Task Implementation Guide](docs/03_Engineering/TASK_IMPLEMENTATION_GUIDE.md) for details.

---

## 🔧 Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Framework** | FastAPI, Uvicorn |
| **Language** | Python 3.12+ |
| **ORM** | SQLAlchemy 2.0 (Async), Alembic |
| **Validation** | Pydantic v2, Pydantic-Settings |
| **Security** | python-jose (JWT), Passlib (bcrypt) |
| **LLM** | Google Gemini API |
| **Embeddings** | Sentence-Transformers, FAISS |
| **Testing** | pytest, pytest-asyncio, HTTPX |
| **Linting** | Ruff, MyPy (strict) |
| **CI/CD** | GitHub Actions |
| **Containers** | Docker, Docker Compose |

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
