"""
Tests for the exception handling system.

Validates that custom exceptions produce correct HTTP status codes
and response structures.
"""

import pytest
from httpx import AsyncClient

from app.core.exceptions import (
    BadRequestException,
    ConflictException,
    ForbiddenException,
    NotFoundException,
    UnauthorizedException,
)


@pytest.mark.unit
class TestCustomExceptions:
    """Tests for custom exception classes."""

    def test_not_found_exception(self) -> None:
        """NotFoundException should have 404 status."""
        exc = NotFoundException("User", "123")
        assert exc.status_code == 404
        assert exc.message == "User not found"
        assert exc.detail == {"resource": "User", "id": "123"}

    def test_conflict_exception(self) -> None:
        """ConflictException should have 409 status."""
        exc = ConflictException("Email already registered")
        assert exc.status_code == 409
        assert exc.message == "Email already registered"

    def test_unauthorized_exception(self) -> None:
        """UnauthorizedException should have 401 status."""
        exc = UnauthorizedException()
        assert exc.status_code == 401
        assert exc.message == "Authentication required"

    def test_forbidden_exception(self) -> None:
        """ForbiddenException should have 403 status."""
        exc = ForbiddenException()
        assert exc.status_code == 403
        assert exc.message == "Insufficient permissions"

    def test_bad_request_exception(self) -> None:
        """BadRequestException should have 400 status."""
        exc = BadRequestException("Invalid input", detail={"field": "email"})
        assert exc.status_code == 400
        assert exc.detail == {"field": "email"}


@pytest.mark.unit
class TestExceptionHandlers:
    """Tests for exception handler integration with FastAPI."""

    async def test_404_response_format(self, client: AsyncClient) -> None:
        """Requesting a non-existent route should return structured 404."""
        response = await client.get("/api/v1/nonexistent")
        # FastAPI returns 404 for unregistered routes
        assert response.status_code == 404
