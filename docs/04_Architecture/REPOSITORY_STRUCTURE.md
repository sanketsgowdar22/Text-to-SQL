# Repository Structure — Enterprise Engineering Framework

> **Version**: 1.0.0
> **Last Updated**: 2026-07-25

---

## Directory Tree

```
Text-to-SQL/
│
├── .github/                                  # GitHub platform configuration
│   ├── ISSUE_TEMPLATE/                       # Issue submission templates
│   │   ├── feature_request.md                # New feature proposal
│   │   ├── bug_report.md                     # Bug report with reproduction
│   │   ├── refactor.md                       # Code refactoring proposal
│   │   ├── documentation.md                  # Documentation improvement
│   │   └── architecture.md                   # Architecture change proposal
│   ├── workflows/                            # CI/CD pipeline definitions
│   │   └── ci.yml                            # Lint + Test + Build pipeline
│   └── PULL_REQUEST_TEMPLATE.md              # Pull request submission template
│
├── docs/                                     # Complete documentation suite
│   ├── audit/                                # Repository audit reports
│   │   ├── PROJECT_AUDIT_REPORT.md           # Initial architecture audit
│   │   └── CURRENT_PROBLEMS.md               # Identified problems & remediation
│   ├── 01_Project/                           # Project-level documentation
│   │   └── README.md                         # Project charter, glossary, team
│   ├── 02_Product/                           # Product requirements
│   │   └── README.md                         # PRD index & feature specs
│   ├── 03_Engineering/                       # Engineering governance
│   │   ├── README.md                         # Engineering docs index
│   │   ├── ENGINEERING_RULES.md              # Mandatory implementation rules
│   │   └── TASK_IMPLEMENTATION_GUIDE.md      # Step-by-step task workflow
│   ├── 04_Architecture/                      # System architecture
│   │   ├── README.md                         # Architecture docs index
│   │   └── REPOSITORY_STRUCTURE.md           # This file
│   ├── 05_Phases/                            # Phase specification documents
│   │   ├── README.md                         # Phase index
│   │   ├── PHASE_00_ENGINEERING_FOUNDATION.md
│   │   ├── PHASE_01_PROJECT_FOUNDATION.md
│   │   ├── PHASE_02_AUTHENTICATION.md
│   │   ├── PHASE_03_DATABASE_CONNECTIONS.md
│   │   ├── PHASE_04_SCHEMA_INGESTION.md
│   │   ├── PHASE_05_EMBEDDINGS.md
│   │   ├── PHASE_06_TEXT_TO_SQL.md
│   │   ├── PHASE_07_GUARDRAILS.md
│   │   └── PHASE_08_FRONTEND_DEPLOYMENT.md
│   ├── 06_Tasks/                             # Task specification documents
│   │   └── README.md                         # Task index & conventions
│   ├── 07_API/                               # API specifications
│   │   └── README.md                         # API docs index
│   ├── 08_AI/                                # AI/ML specifications
│   │   └── README.md                         # AI pipeline docs index
│   ├── 09_Database/                          # Database documentation
│   │   └── README.md                         # Database docs index
│   ├── 10_Testing/                           # Testing documentation
│   │   └── README.md                         # Testing strategy & standards
│   ├── 11_Deployment/                        # Deployment documentation
│   │   └── README.md                         # Deployment & infrastructure
│   ├── 12_Security/                          # Security documentation
│   │   └── README.md                         # Security policies & practices
│   ├── 13_Git/                               # Git workflow documentation
│   │   ├── README.md                         # Git docs index
│   │   └── GIT_WORKFLOW.md                   # Complete Git strategy
│   ├── 14_Development/                       # Developer documentation
│   │   └── README.md                         # Onboarding & local setup
│   ├── 15_Standards/                         # Coding standards
│   │   └── README.md                         # Standards index
│   ├── 16_Templates/                         # Reusable document templates
│   │   ├── README.md                         # Template index
│   │   ├── PHASE_TEMPLATE.md
│   │   ├── TASK_TEMPLATE.md
│   │   ├── API_SPECIFICATION_TEMPLATE.md
│   │   ├── ADR_TEMPLATE.md
│   │   ├── TESTING_REPORT_TEMPLATE.md
│   │   ├── REVIEW_REPORT_TEMPLATE.md
│   │   └── RELEASE_REPORT_TEMPLATE.md
│   ├── 17_Reviews/                           # Review process documentation
│   │   └── README.md                         # Review process & checklists
│   ├── 18_Releases/                          # Release process documentation
│   │   └── README.md                         # Release process & versioning
│   ├── 19_Decisions/                         # Architecture Decision Records
│   │   └── README.md                         # ADR index
│   ├── architecture.md                       # [PRESERVED] Original architecture doc
│   ├── prd_and_roadmap.md                    # [PRESERVED] Original PRD
│   ├── database_schema.md                    # [PRESERVED] Original schema doc
│   ├── api_specification.md                  # [PRESERVED] Original API spec
│   └── task_wise_implementation_plan.md      # [PRESERVED] Original task plan
│
├── backend/                                  # FastAPI Backend Service
│   ├── alembic/                              # Database migration scripts
│   │   ├── versions/                         # Migration version files
│   │   ├── env.py                            # Alembic environment config
│   │   └── script.py.mako                    # Migration template
│   ├── app/                                  # Application source code
│   │   ├── core/                             # Core infrastructure
│   │   │   ├── config.py                     # Pydantic settings
│   │   │   ├── constants.py                  # Application constants
│   │   │   ├── database.py                   # SQLAlchemy engine & sessions
│   │   │   ├── exceptions.py                 # Exception hierarchy & handlers
│   │   │   ├── middleware.py                 # CORS, logging, request ID
│   │   │   └── security.py                  # Password hashing, JWT utils
│   │   ├── features/                         # Modular feature packages
│   │   │   ├── auth/                         # Authentication & user management
│   │   │   ├── connections/                  # Database connection management
│   │   │   ├── schema/                       # Schema introspection engine
│   │   │   ├── embeddings/                   # Vector embeddings & RAG
│   │   │   ├── query/                        # Text-to-SQL generation
│   │   │   └── execution/                    # SQL safety & execution
│   │   ├── shared/                           # Shared utilities & base models
│   │   │   ├── models.py                     # ORM mixins (UUID, Timestamp)
│   │   │   ├── schemas.py                    # Response wrappers
│   │   │   ├── pagination.py                 # Pagination helpers
│   │   │   ├── validators.py                 # Input validators
│   │   │   └── utils.py                      # Common utilities
│   │   └── main.py                           # FastAPI application factory
│   ├── tests/                                # Test suite
│   │   ├── conftest.py                       # Shared fixtures
│   │   └── unit/                             # Unit tests
│   ├── .env.example                          # Environment template
│   ├── alembic.ini                           # Alembic configuration
│   ├── pyproject.toml                        # Project & tool configuration
│   ├── requirements.txt                      # Production dependencies
│   ├── requirements-dev.txt                  # Development dependencies
│   └── README.md                             # Backend quickstart
│
├── frontend/                                 # Frontend Application (Phase 8)
│   └── README.md                             # Architecture placeholder
│
├── ai/                                       # AI/ML Service Layer
│   └── README.md                             # Architecture placeholder
│
├── database/                                 # Database Management
│   └── README.md                             # Scripts & migration guide
│
├── shared/                                   # Cross-Service Contracts
│   └── README.md                             # Shared interfaces & types
│
├── scripts/                                  # DevOps & Utility Scripts
│   └── README.md                             # Script index
│
├── docker/                                   # Container Configuration
│   ├── Dockerfile.backend                    # Backend multi-stage build
│   └── docker-compose.yml                    # Service orchestration
│
├── tests/                                    # Top-Level Integration/E2E Tests
│   └── README.md                             # Cross-service testing
│
├── .gitignore                                # Git ignore rules
├── CHANGELOG.md                              # Version changelog
├── CODE_OF_CONDUCT.md                        # Contributor behavior standards
├── CONTRIBUTING.md                           # Contribution guidelines
├── LICENSE                                   # Apache 2.0 license
└── README.md                                 # Project root documentation
```

---

## Design Principles

1. **Separation of Concerns**: Each top-level directory serves exactly one purpose.
2. **Documentation as First-Class Citizen**: The `docs/` hierarchy mirrors the complexity of the project itself.
3. **Service Isolation**: Each service (backend, frontend, AI) has its own directory with independent lifecycle.
4. **Progressive Disclosure**: Numbered documentation folders create a natural reading order.
5. **Template-Driven Development**: Reusable templates ensure consistency across all project artifacts.
6. **Backwards Compatibility**: Original documentation files are preserved at their existing paths.
