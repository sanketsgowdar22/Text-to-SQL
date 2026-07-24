"""
Tests for the exception handling system.

Validates that custom exceptions produce correct HTTP status codes
and response structures.
"""

import pytest
from httpx import AsyncClient

from app.core.exceptions import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    UnauthorizedError,
)


@pytest.mark.unit
class TestCustomErrors:
    """Tests for custom exception classes."""

    def test_not_found_exception(self) -> None:
        """NotFoundError should have 404 status."""
        exc = NotFoundError("User", "123")
        assert exc.status_code == 404
        assert exc.message == "User not found"
        assert exc.detail == {"resource": "User", "id": "123"}

    def test_conflict_exception(self) -> None:
        """ConflictError should have 409 status."""
        exc = ConflictError("Email already registered")
        assert exc.status_code == 409
        assert exc.message == "Email already registered"

    def test_unauthorized_exception(self) -> None:
        """UnauthorizedError should have 401 status."""
        exc = UnauthorizedError()
        assert exc.status_code == 401
        assert exc.message == "Authentication required"

    def test_forbidden_exception(self) -> None:
        """ForbiddenError should have 403 status."""
        exc = ForbiddenError()
        assert exc.status_code == 403
        assert exc.message == "Insufficient permissions"

    def test_bad_request_exception(self) -> None:
        """BadRequestError should have 400 status."""
        exc = BadRequestError("Invalid input", detail={"field": "email"})
        assert exc.status_code == 400
        assert exc.detail == {"field": "email"}


@pytest.mark.unit
class TestErrorHandlers:
    """Tests for exception handler integration with FastAPI."""

    async def test_404_response_format(self, client: AsyncClient) -> None:
        """Requesting a non-existent route should return structured 404."""
        response = await client.get("/api/v1/nonexistent")
        # FastAPI returns 404 for unregistered routes
        assert response.status_code == 404
