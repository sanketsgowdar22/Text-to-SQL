"""
Unit tests for Auth Service layer.

Tests registration, login, token refresh, and profile operations
using an in-memory test database.
"""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError, UnauthorizedError
from app.core.security import create_refresh_token, decode_token
from app.features.auth.schemas import RegisterRequest, UpdateProfileRequest
from app.features.auth.service import AuthService


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def auth_service(db_session: AsyncSession) -> AuthService:
    """Provide an AuthService bound to the test database session."""
    return AuthService(db_session)


async def _register_user(
    service: AuthService,
    email: str = "test@example.com",
    password: str = "SecurePass123!",
    full_name: str = "Test User",
):
    """Helper to register a user and return the AuthResponse."""
    data = RegisterRequest(email=email, password=password, full_name=full_name)
    return await service.register(data)


# ── Registration Tests ───────────────────────────────────────────────────────


class TestRegister:
    """Test user registration."""

    @pytest.mark.asyncio
    async def test_register_success(self, auth_service: AuthService):
        """Successful registration returns tokens and user profile."""
        result = await _register_user(auth_service)

        assert result.user.email == "test@example.com"
        assert result.user.full_name == "Test User"
        assert result.user.is_active is True
        assert result.user.is_superuser is False
        assert result.tokens.access_token
        assert result.tokens.refresh_token
        assert result.tokens.token_type == "bearer"

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, auth_service: AuthService):
        """Registration with an existing email raises ConflictError."""
        await _register_user(auth_service)

        with pytest.raises(ConflictError, match="already exists"):
            await _register_user(auth_service)

    @pytest.mark.asyncio
    async def test_register_tokens_are_valid_jwt(self, auth_service: AuthService):
        """Tokens returned on registration are valid JWTs."""
        result = await _register_user(auth_service)

        access_payload = decode_token(result.tokens.access_token)
        assert access_payload["type"] == "access"
        assert access_payload["sub"] == result.user.id

        refresh_payload = decode_token(result.tokens.refresh_token)
        assert refresh_payload["type"] == "refresh"
        assert refresh_payload["sub"] == result.user.id


# ── Login Tests ──────────────────────────────────────────────────────────────


class TestLogin:
    """Test user login."""

    @pytest.mark.asyncio
    async def test_login_success(self, auth_service: AuthService):
        """Login with correct credentials returns tokens and profile."""
        await _register_user(auth_service)
        result = await auth_service.login("test@example.com", "SecurePass123!")

        assert result.user.email == "test@example.com"
        assert result.tokens.access_token
        assert result.tokens.refresh_token

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, auth_service: AuthService):
        """Login with wrong password raises UnauthorizedError."""
        await _register_user(auth_service)

        with pytest.raises(UnauthorizedError, match="Invalid email or password"):
            await auth_service.login("test@example.com", "WrongPassword!")

    @pytest.mark.asyncio
    async def test_login_nonexistent_email(self, auth_service: AuthService):
        """Login with unknown email raises UnauthorizedError."""
        with pytest.raises(UnauthorizedError, match="Invalid email or password"):
            await auth_service.login("nobody@example.com", "Whatever123!")

    @pytest.mark.asyncio
    async def test_login_inactive_user(self, auth_service: AuthService, db_session: AsyncSession):
        """Login with a disabled account raises UnauthorizedError."""
        result = await _register_user(auth_service)

        # Deactivate user directly
        user = await auth_service._get_user_by_id(result.user.id)
        assert user is not None
        user.is_active = False
        await db_session.flush()

        with pytest.raises(UnauthorizedError, match="disabled"):
            await auth_service.login("test@example.com", "SecurePass123!")


# ── Token Refresh Tests ──────────────────────────────────────────────────────


class TestRefreshToken:
    """Test token refresh flow."""

    @pytest.mark.asyncio
    async def test_refresh_success(self, auth_service: AuthService):
        """Valid refresh token returns new token pair."""
        result = await _register_user(auth_service)
        new_tokens = await auth_service.refresh_token(result.tokens.refresh_token)

        assert new_tokens.access_token
        assert new_tokens.refresh_token
        # Both tokens should be valid JWTs for the same user
        new_payload = decode_token(new_tokens.access_token)
        old_payload = decode_token(result.tokens.access_token)
        assert new_payload["sub"] == old_payload["sub"]
        assert new_payload["type"] == "access"

    @pytest.mark.asyncio
    async def test_refresh_with_access_token_fails(self, auth_service: AuthService):
        """Using an access token for refresh raises UnauthorizedError."""
        result = await _register_user(auth_service)

        with pytest.raises(UnauthorizedError, match="Invalid token type"):
            await auth_service.refresh_token(result.tokens.access_token)

    @pytest.mark.asyncio
    async def test_refresh_with_invalid_token(self, auth_service: AuthService):
        """Invalid JWT string raises UnauthorizedError."""
        with pytest.raises(UnauthorizedError, match="Invalid or expired"):
            await auth_service.refresh_token("not-a-valid-token")

    @pytest.mark.asyncio
    async def test_refresh_nonexistent_user(self, auth_service: AuthService):
        """Refresh token for a deleted user raises UnauthorizedError."""
        # Create a refresh token for a non-existent user ID
        fake_token = create_refresh_token(subject="nonexistent-user-id")

        with pytest.raises(UnauthorizedError, match="User not found"):
            await auth_service.refresh_token(fake_token)


# ── Profile Tests ────────────────────────────────────────────────────────────


class TestProfile:
    """Test profile get and update."""

    @pytest.mark.asyncio
    async def test_get_profile(self, auth_service: AuthService):
        """Get profile returns the correct user data."""
        result = await _register_user(auth_service)
        profile = await auth_service.get_profile(result.user.id)

        assert profile.email == "test@example.com"
        assert profile.full_name == "Test User"

    @pytest.mark.asyncio
    async def test_get_profile_not_found(self, auth_service: AuthService):
        """Get profile for non-existent ID raises NotFoundError."""
        with pytest.raises(NotFoundError):
            await auth_service.get_profile("nonexistent-id")

    @pytest.mark.asyncio
    async def test_update_profile_name(self, auth_service: AuthService):
        """Updating full_name changes it."""
        result = await _register_user(auth_service)
        data = UpdateProfileRequest(full_name="New Name")
        updated = await auth_service.update_profile(result.user.id, data)

        assert updated.full_name == "New Name"
        assert updated.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_update_profile_password(self, auth_service: AuthService):
        """Updating password allows login with new password."""
        result = await _register_user(auth_service)
        data = UpdateProfileRequest(password="NewSecurePass456!")
        await auth_service.update_profile(result.user.id, data)

        # Login with new password should work
        login_result = await auth_service.login("test@example.com", "NewSecurePass456!")
        assert login_result.user.id == result.user.id

        # Login with old password should fail
        with pytest.raises(UnauthorizedError):
            await auth_service.login("test@example.com", "SecurePass123!")

    @pytest.mark.asyncio
    async def test_update_profile_not_found(self, auth_service: AuthService):
        """Updating non-existent user raises NotFoundError."""
        data = UpdateProfileRequest(full_name="Ghost")
        with pytest.raises(NotFoundError):
            await auth_service.update_profile("nonexistent-id", data)
