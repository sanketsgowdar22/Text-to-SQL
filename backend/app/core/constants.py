"""
Application constants.

Centralized location for all application-wide constants.
Avoid scattering magic strings and numbers throughout the codebase.
"""

# ── API Versioning ───────────────────────────────────────────────────────────
API_V1_PREFIX = "/api/v1"

# ── Database ─────────────────────────────────────────────────────────────────
DB_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# ── Pagination ───────────────────────────────────────────────────────────────
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MIN_PAGE_SIZE = 1

# ── Query Limits ─────────────────────────────────────────────────────────────
MAX_QUERY_LENGTH = 5000
MAX_SQL_LENGTH = 10000
MAX_RESULT_ROWS = 1000
QUERY_TIMEOUT_SECONDS = 30

# ── Token Limits ─────────────────────────────────────────────────────────────
MAX_PROMPT_TOKENS = 4000
MAX_RESPONSE_TOKENS = 2000

# ── Connection Limits ────────────────────────────────────────────────────────
MAX_CONNECTIONS_PER_USER = 10
CONNECTION_TEST_TIMEOUT_SECONDS = 10

# ── Supported Database Types ─────────────────────────────────────────────────
SUPPORTED_DB_TYPES = frozenset({"postgresql", "mysql", "sqlite"})

# ── User Roles ───────────────────────────────────────────────────────────────
ROLE_ADMIN = "admin"
ROLE_USER = "user"
ROLE_VIEWER = "viewer"
VALID_ROLES = frozenset({ROLE_ADMIN, ROLE_USER, ROLE_VIEWER})

# ── HTTP Headers ─────────────────────────────────────────────────────────────
HEADER_REQUEST_ID = "X-Request-ID"
