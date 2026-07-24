# 14 — Development Documentation

Developer onboarding, local environment setup, and development workflow.

---

## Local Development Setup

### Prerequisites

- Python 3.11+
- Git

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
copy .env.example .env         # Windows
# cp .env.example .env         # Linux/Mac

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
cd backend
.\venv\Scripts\python.exe -m pytest -v
```

### Code Quality

```bash
# Linting
ruff check app/ tests/

# Formatting
ruff format app/ tests/

# Type checking
mypy app/
```

---

## Development Workflow

Before writing any code, read:
1. [Engineering Rules](../03_Engineering/ENGINEERING_RULES.md)
2. [Task Implementation Guide](../03_Engineering/TASK_IMPLEMENTATION_GUIDE.md)
3. [Git Workflow](../13_Git/GIT_WORKFLOW.md)

---

## Interactive API Documentation

Once the backend is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Related Documentation

- [Coding Standards](../15_Standards/README.md)
- [Testing Strategy](../10_Testing/README.md)
- [Architecture Overview](../../docs/architecture.md)
