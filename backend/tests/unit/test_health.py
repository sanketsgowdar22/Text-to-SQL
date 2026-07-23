"""
Tests for the health check endpoints.

Validates that the health and readiness endpoints return correct
status codes and response structures.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    async def test_health_returns_200(self, client: AsyncClient) -> None:
        """Health check should return 200 with healthy status."""
        response = await client.get("/api/v1/health")
        assert response.status_code == 200

    async def test_health_response_structure(self, client: AsyncClient) -> None:
        """Health response should have correct JSON structure."""
        response = await client.get("/api/v1/health")
        data = response.json()

        assert data["success"] is True
        assert "data" in data
        assert data["data"]["status"] == "healthy"
        assert "version" in data["data"]
        assert "environment" in data["data"]
        assert "timestamp" in data["data"]

    async def test_health_returns_version(self, client: AsyncClient) -> None:
        """Health response should include the application version."""
        response = await client.get("/api/v1/health")
        data = response.json()
        assert data["data"]["version"] == "0.1.0"

    async def test_health_returns_message(self, client: AsyncClient) -> None:
        """Health response should include a success message."""
        response = await client.get("/api/v1/health")
        data = response.json()
        assert data["message"] == "Service is healthy"


@pytest.mark.unit
class TestReadinessEndpoint:
    """Tests for GET /api/v1/health/ready."""

    async def test_readiness_returns_200(self, client: AsyncClient) -> None:
        """Readiness probe should return 200 when system is ready."""
        response = await client.get("/api/v1/health/ready")
        assert response.status_code == 200

    async def test_readiness_response_structure(self, client: AsyncClient) -> None:
        """Readiness response should include check results."""
        response = await client.get("/api/v1/health/ready")
        data = response.json()

        assert data["success"] is True
        assert data["data"]["ready"] is True
        assert "checks" in data["data"]
        assert data["data"]["checks"]["database"] == "ok"
