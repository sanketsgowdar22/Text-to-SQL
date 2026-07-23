"""
Tests for shared validators.

Validates email, password, database name, and string sanitization.
"""

import pytest

from app.shared.validators import (
    is_valid_database_name,
    is_valid_email,
    sanitize_string,
    validate_password_strength,
)


@pytest.mark.unit
class TestEmailValidator:
    """Tests for email validation."""

    @pytest.mark.parametrize(
        "email",
        [
            "user@example.com",
            "test.user@domain.org",
            "user+tag@company.co",
            "first.last@sub.domain.com",
        ],
    )
    def test_valid_emails(self, email: str) -> None:
        """Valid email addresses should pass validation."""
        assert is_valid_email(email) is True

    @pytest.mark.parametrize(
        "email",
        [
            "",
            "not-an-email",
            "@domain.com",
            "user@",
            "user@.com",
            "user@domain",
            "user space@domain.com",
        ],
    )
    def test_invalid_emails(self, email: str) -> None:
        """Invalid email addresses should fail validation."""
        assert is_valid_email(email) is False


@pytest.mark.unit
class TestPasswordValidator:
    """Tests for password strength validation."""

    def test_strong_password(self) -> None:
        """A strong password should return no violations."""
        violations = validate_password_strength("SecureP@ss1")
        assert violations == []

    def test_too_short(self) -> None:
        """Password under 8 characters should fail."""
        violations = validate_password_strength("Ab1!")
        assert any("at least 8" in v for v in violations)

    def test_no_uppercase(self) -> None:
        """Password without uppercase should fail."""
        violations = validate_password_strength("password1!")
        assert any("uppercase" in v for v in violations)

    def test_no_lowercase(self) -> None:
        """Password without lowercase should fail."""
        violations = validate_password_strength("PASSWORD1!")
        assert any("lowercase" in v for v in violations)

    def test_no_digit(self) -> None:
        """Password without digit should fail."""
        violations = validate_password_strength("Password!!")
        assert any("digit" in v for v in violations)

    def test_no_special_char(self) -> None:
        """Password without special character should fail."""
        violations = validate_password_strength("Password12")
        assert any("special" in v for v in violations)


@pytest.mark.unit
class TestDatabaseNameValidator:
    """Tests for database name validation."""

    @pytest.mark.parametrize(
        "name",
        ["mydb", "My_Database", "prod-db-01", "a"],
    )
    def test_valid_names(self, name: str) -> None:
        """Valid database names should pass."""
        assert is_valid_database_name(name) is True

    @pytest.mark.parametrize(
        "name",
        ["", "1startswithnumber", "-startswithdash", "has spaces", "a" * 65],
    )
    def test_invalid_names(self, name: str) -> None:
        """Invalid database names should fail."""
        assert is_valid_database_name(name) is False


@pytest.mark.unit
class TestSanitizeString:
    """Tests for string sanitization."""

    def test_strips_whitespace(self) -> None:
        """Should strip leading and trailing whitespace."""
        assert sanitize_string("  hello  ") == "hello"

    def test_removes_null_bytes(self) -> None:
        """Should remove null bytes."""
        assert sanitize_string("hello\x00world") == "helloworld"

    def test_normal_string_unchanged(self) -> None:
        """Normal strings should pass through unchanged."""
        assert sanitize_string("hello world") == "hello world"
