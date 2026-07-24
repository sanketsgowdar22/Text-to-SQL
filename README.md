# AI Data Intelligence Platform (ADIP) — Text-to-SQL

Enterprise AI Data Intelligence Platform powered by FastAPI, Google Gemini LLM, Sentence-Transformers, FAISS, and SQLAlchemy.

---

## 📚 Technical Documentation Suite (Phase 0)

| Document | Description |
| :--- | :--- |
| **[System Architecture](docs/architecture.md)** | Multi-tier topology, component breakdown, security model, and technology stack. |
| **[PRD & Roadmap](docs/prd_and_roadmap.md)** | Product vision, functional requirements, and 8-phase milestone roadmap. |
| **[Database Schema](docs/database_schema.md)** | System database ERD, data models (`users`, `db_connections`, `schema_metadata`, `query_logs`). |
| **[API Specification](docs/api_specification.md)** | RESTful API endpoint contracts, schemas, and error structures. |
| **[Task Implementation Plan](docs/task_wise_implementation_plan.md)** | Task-by-task execution specification for all 8 project phases. |

---

## 🚀 Quick Start (Backend Service)

### Setup & Installation

```bash
# Change directory to backend
cd backend

# Create & activate Python virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy environment variables file
copy .env.example .env
```

### Running Tests

```bash
# Run pytest using virtual environment python
.\venv\Scripts\python.exe -m pytest
```

### Running Development Server

```bash
# Start FastAPI application with auto-reload
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Once running:
- **Interactive OpenAPI Documentation**: `http://localhost:8000/docs`
- **ReDoc API View**: `http://localhost:8000/redoc`

---

## 🛠️ Project Structure

```
Text-to-SQL/
├── docs/                      # Phase 0 System Documentation Suite
│   ├── api_specification.md   # API Contracts & Payloads
│   ├── architecture.md        # System Design & Security Topology
│   ├── database_schema.md     # Relational ERD & Data Models
│   ├── prd_and_roadmap.md     # Requirements & Phase Roadmap
│   └── task_wise_implementation_plan.md  # Granular Task Execution Plan
├── backend/                   # FastAPI Backend Application
│   ├── alembic/               # Database Migration Scripts
│   ├── app/
│   │   ├── core/              # Config, Security, DB, Exception Handlers
│   │   ├── features/          # Modular Feature Routers (Auth, Connections, Schema, Query)
│   │   ├── shared/            # Common Models, Utilities & Schemas
│   │   └── main.py            # FastAPI Application Factory
│   ├── tests/                 # Unit & Integration Test Suite
│   └── pyproject.toml         # Tooling & Pytest Configuration
└── README.md                  # Project Root Documentation Page
```
