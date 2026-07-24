# Contributing to ADIP

Thank you for your interest in contributing to the AI Data Intelligence Platform!

---

## Before You Start

1. **Read the Engineering Rules**: [docs/03_Engineering/ENGINEERING_RULES.md](docs/03_Engineering/ENGINEERING_RULES.md)
2. **Read the Task Implementation Guide**: [docs/03_Engineering/TASK_IMPLEMENTATION_GUIDE.md](docs/03_Engineering/TASK_IMPLEMENTATION_GUIDE.md)
3. **Read the Git Workflow**: [docs/13_Git/GIT_WORKFLOW.md](docs/13_Git/GIT_WORKFLOW.md)

---

## Development Workflow

### 1. Set Up Your Environment

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
copy .env.example .env
```

### 2. Find or Create a Task

- Check [docs/05_Phases/](docs/05_Phases/) for the current phase
- Check [docs/06_Tasks/](docs/06_Tasks/) for existing task documents
- If no task document exists, create one using [TASK_TEMPLATE.md](docs/16_Templates/TASK_TEMPLATE.md)

### 3. Create a Feature Branch

```bash
git checkout development
git pull origin development
git checkout -b feature/<phase>-<task-id>-<description>
```

### 4. Implement

- Follow [Coding Standards](docs/15_Standards/README.md)
- Write tests for all new functionality
- Update documentation in the same PR

### 5. Submit a Pull Request

- Use the [PR Template](.github/PULL_REQUEST_TEMPLATE.md)
- Fill in ALL sections
- Ensure CI pipeline passes

---

## Code Quality Requirements

| Tool | Requirement |
|------|-------------|
| **Ruff** | All lint checks pass |
| **MyPy** | Strict type checking passes |
| **pytest** | All tests pass, coverage ≥ 80% |

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(auth): implement user registration endpoint
fix(database): resolve connection pool leak
docs(phase-02): add authentication phase document
test(auth): add JWT expiration edge case tests
```

---

## Reporting Issues

Use the appropriate [Issue Template](.github/ISSUE_TEMPLATE/):
- **Feature Request** — New functionality proposals
- **Bug Report** — Bug reports with reproduction steps
- **Refactor** — Code improvement proposals
- **Documentation** — Documentation improvements
- **Architecture** — Architecture change proposals

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to abide by its terms.
