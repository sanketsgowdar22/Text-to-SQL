"""
Integration tests for Auth Router endpoints.

Tests all auth endpoints via the HTTPX async test client,
validating request/response contracts, status codes, and error handling.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.core.constants import API_V1_PREFIX

AUTH_PREFIX = f"{API_V1_PREFIX}/auth"


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _register(client: AsyncClient, **overrides) -> dict:
    """Helper to register a user and return the full response JSON."""
    payload = {
        "email": "test@example.com",
        "password": "SecurePass123!",
        "full_name": "Test User",
        **overrides,
    }
    resp = await client.post(f"{AUTH_PREFIX}/register", json=payload)
    return resp.json(), resp.status_code


async def _login(client: AsyncClient, email: str = "test@example.com", password: str = "SecurePass123!"):
    """Helper to login and return response JSON + status code."""
    resp = await client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": email, "password": password},
    )
    return resp.json(), resp.status_code


def _auth_header(access_token: str) -> dict:
    """Build an Authorization header."""
    return {"Authorization": f"Bearer {access_token}"}


# ── POST /register ──────────────────────────────────────────────────────────


class TestRegisterEndpoint:
    """Test POST /auth/register."""

    @pytest.mark.asyncio
    async def test_register_success(self, client: AsyncClient):
        body, status = await _register(client)

        assert status == 201
        assert body["success"] is True
        assert body["data"]["user"]["email"] == "test@example.com"
        assert body["data"]["tokens"]["access_token"]
        assert body["data"]["tokens"]["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, client: AsyncClient):
        await _register(client)
        body, status = await _register(client)

        assert status == 409
        assert body["success"] is False
        assert "already exists" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_register_invalid_email(self, client: AsyncClient):
        body, status = await _register(client, email="not-an-email")

        assert status == 422
        assert body["success"] is False

    @pytest.mark.asyncio
    async def test_register_short_password(self, client: AsyncClient):
        body, status = await _register(client, password="short")

        assert status == 422

    @pytest.mark.asyncio
    async def test_register_missing_fields(self, client: AsyncClient):
        resp = await client.post(f"{AUTH_PREFIX}/register", json={})
        assert resp.status_code == 422


# ── POST /login ──────────────────────────────────────────────────────────────


class TestLoginEndpoint:
    """Test POST /auth/login."""

    @pytest.mark.asyncio
    async def test_login_success(self, client: AsyncClient):
        await _register(client)
        body, status = await _login(client)

        assert status == 200
        assert body["success"] is True
        assert body["data"]["user"]["email"] == "test@example.com"
        assert body["data"]["tokens"]["access_token"]

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client: AsyncClient):
        await _register(client)
        body, status = await _login(client, password="WrongPassword!")

        assert status == 401
        assert body["success"] is False

    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, client: AsyncClient):
        body, status = await _login(client, email="nobody@example.com")

        assert status == 401


# ── POST /refresh ────────────────────────────────────────────────────────────


class TestRefreshEndpoint:
    """Test POST /auth/refresh."""

    @pytest.mark.asyncio
    async def test_refresh_success(self, client: AsyncClient):
        reg_body, _ = await _register(client)
        refresh_token = reg_body["data"]["tokens"]["refresh_token"]

        resp = await client.post(
            f"{AUTH_PREFIX}/refresh",
            json={"refresh_token": refresh_token},
        )
        body = resp.json()

        assert resp.status_code == 200
        assert body["data"]["access_token"]
        assert body["data"]["refresh_token"]

    @pytest.mark.asyncio
    async def test_refresh_with_invalid_token(self, client: AsyncClient):
        resp = await client.post(
            f"{AUTH_PREFIX}/refresh",
            json={"refresh_token": "garbage-token"},
        )

        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_with_access_token(self, client: AsyncClient):
        reg_body, _ = await _register(client)
        access_token = reg_body["data"]["tokens"]["access_token"]

        resp = await client.post(
            f"{AUTH_PREFIX}/refresh",
            json={"refresh_token": access_token},
        )

        assert resp.status_code == 401


# ── GET /me ──────────────────────────────────────────────────────────────────


class TestGetProfileEndpoint:
    """Test GET /auth/me."""

    @pytest.mark.asyncio
    async def test_get_profile_success(self, client: AsyncClient):
        reg_body, _ = await _register(client)
        token = reg_body["data"]["tokens"]["access_token"]

        resp = await client.get(f"{AUTH_PREFIX}/me", headers=_auth_header(token))
        body = resp.json()

        assert resp.status_code == 200
        assert body["data"]["email"] == "test@example.com"
        assert body["data"]["full_name"] == "Test User"

    @pytest.mark.asyncio
    async def test_get_profile_no_token(self, client: AsyncClient):
        resp = await client.get(f"{AUTH_PREFIX}/me")

        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_get_profile_invalid_token(self, client: AsyncClient):
        resp = await client.get(
            f"{AUTH_PREFIX}/me",
            headers=_auth_header("invalid-token"),
        )

        assert resp.status_code == 401


# ── PUT /me ──────────────────────────────────────────────────────────────────


class TestUpdateProfileEndpoint:
    """Test PUT /auth/me."""

    @pytest.mark.asyncio
    async def test_update_name(self, client: AsyncClient):
        reg_body, _ = await _register(client)
        token = reg_body["data"]["tokens"]["access_token"]

        resp = await client.put(
            f"{AUTH_PREFIX}/me",
            headers=_auth_header(token),
            json={"full_name": "Updated Name"},
        )
        body = resp.json()

        assert resp.status_code == 200
        assert body["data"]["full_name"] == "Updated Name"

    @pytest.mark.asyncio
    async def test_update_password_then_login(self, client: AsyncClient):
        reg_body, _ = await _register(client)
        token = reg_body["data"]["tokens"]["access_token"]

        # Change password
        resp = await client.put(
            f"{AUTH_PREFIX}/me",
            headers=_auth_header(token),
            json={"password": "NewSecurePass456!"},
        )
        assert resp.status_code == 200

        # Login with new password
        _, status = await _login(client, password="NewSecurePass456!")
        assert status == 200

        # Old password fails
        _, status = await _login(client, password="SecurePass123!")
        assert status == 401

    @pytest.mark.asyncio
    async def test_update_profile_no_auth(self, client: AsyncClient):
        resp = await client.put(
            f"{AUTH_PREFIX}/me",
            json={"full_name": "Hacker"},
        )

        assert resp.status_code == 401
