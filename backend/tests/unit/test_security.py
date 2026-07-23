"""
Tests for the security module.

Validates password hashing, JWT token creation, and token verification.
"""

import pytest

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


@pytest.mark.unit
class TestPasswordHashing:
    """Tests for password hashing and verification."""

    def test_hash_password_returns_hash(self) -> None:
        """hash_password should return a bcrypt hash string."""
        hashed = hash_password("SecureP@ss1")
        assert hashed != "SecureP@ss1"
        assert hashed.startswith("$2b$")

    def test_verify_correct_password(self) -> None:
        """verify_password should return True for correct password."""
        hashed = hash_password("SecureP@ss1")
        assert verify_password("SecureP@ss1", hashed) is True

    def test_verify_incorrect_password(self) -> None:
        """verify_password should return False for wrong password."""
        hashed = hash_password("SecureP@ss1")
        assert verify_password("WrongPassword1!", hashed) is False

    def test_different_hashes_for_same_password(self) -> None:
        """Two hashes of the same password should differ (unique salts)."""
        hash1 = hash_password("SecureP@ss1")
        hash2 = hash_password("SecureP@ss1")
        assert hash1 != hash2


@pytest.mark.unit
class TestJWTTokens:
    """Tests for JWT token creation and verification."""

    def test_create_access_token(self) -> None:
        """create_access_token should return a JWT string."""
        token = create_access_token(subject="user-123")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_decode_access_token(self) -> None:
        """decode_token should return the correct subject."""
        token = create_access_token(subject="user-123")
        payload = decode_token(token)
        assert payload["sub"] == "user-123"
        assert payload["type"] == "access"

    def test_access_token_with_extra_claims(self) -> None:
        """Extra claims should be included in the token payload."""
        token = create_access_token(
            subject="user-123",
            extra_claims={"role": "admin"},
        )
        payload = decode_token(token)
        assert payload["role"] == "admin"

    def test_create_refresh_token(self) -> None:
        """create_refresh_token should return a JWT with refresh type."""
        token = create_refresh_token(subject="user-123")
        payload = decode_token(token)
        assert payload["sub"] == "user-123"
        assert payload["type"] == "refresh"

    def test_token_contains_timestamps(self) -> None:
        """Token should contain iat and exp timestamps."""
        token = create_access_token(subject="user-123")
        payload = decode_token(token)
        assert "iat" in payload
        assert "exp" in payload
        assert payload["exp"] > payload["iat"]
