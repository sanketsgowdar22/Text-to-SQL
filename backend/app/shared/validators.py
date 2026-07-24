"""
Shared validators for input validation across features.

Provides reusable validation functions for common patterns
like emails, passwords, database connection strings, etc.
"""

import re

# ── Email Validation ─────────────────────────────────────────────────────────

_EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def is_valid_email(email: str) -> bool:
    """
    Validate an email address format.

    Args:
        email: The email address to validate.

    Returns:
        True if the email format is valid.
    """
    return bool(_EMAIL_REGEX.match(email))


# ── Password Validation ─────────────────────────────────────────────────────

_PASSWORD_MIN_LENGTH = 8
_PASSWORD_MAX_LENGTH = 128


def validate_password_strength(password: str) -> list[str]:
    """
    Validate password strength and return a list of violations.

    Requirements:
    - At least 8 characters
    - At most 128 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character

    Args:
        password: The password to validate.

    Returns:
        A list of violation messages. Empty list means the password is valid.
    """
    violations: list[str] = []

    if len(password) < _PASSWORD_MIN_LENGTH:
        violations.append(f"Password must be at least {_PASSWORD_MIN_LENGTH} characters")
    if len(password) > _PASSWORD_MAX_LENGTH:
        violations.append(f"Password must be at most {_PASSWORD_MAX_LENGTH} characters")
    if not re.search(r"[A-Z]", password):
        violations.append("Password must contain at least one uppercase letter")
    if not re.search(r"[a-z]", password):
        violations.append("Password must contain at least one lowercase letter")
    if not re.search(r"\d", password):
        violations.append("Password must contain at least one digit")
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password):
        violations.append("Password must contain at least one special character")

    return violations


# ── String Sanitization ──────────────────────────────────────────────────────


def sanitize_string(value: str) -> str:
    """
    Basic string sanitization: strip whitespace and remove null bytes.

    Args:
        value: The string to sanitize.

    Returns:
        The sanitized string.
    """
    return value.strip().replace("\x00", "")


# ── Database Name Validation ────────────────────────────────────────────────

_DB_NAME_REGEX = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")


def is_valid_database_name(name: str) -> bool:
    """
    Validate a database/connection name.

    Must start with a letter, contain only alphanumeric characters,
    hyphens, and underscores, and be at most 64 characters.

    Args:
        name: The database name to validate.

    Returns:
        True if the name is valid.
    """
    return bool(_DB_NAME_REGEX.match(name))
