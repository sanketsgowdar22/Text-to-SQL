"""
Auth service layer.

Contains all business logic for authentication operations:
register, authenticate, refresh tokens, get/update profile.
"""

import logging

from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError, UnauthorizedError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.features.auth.models import User
from app.features.auth.schemas import (
    AuthResponse,
    RegisterRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserResponse,
)

logger = logging.getLogger(__name__)


class AuthService:
    """Authentication and user management service."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ── User Registration ────────────────────────────────────────────────

    async def register(self, data: RegisterRequest) -> AuthResponse:
        """
        Register a new user account.

        Args:
            data: Registration request with email, password, full_name.

        Returns:
            AuthResponse with tokens and user profile.

        Raises:
            ConflictError: If the email is already registered.
        """
        # Check for existing user
        existing = await self._get_user_by_email(data.email)
        if existing:
            raise ConflictError("A user with this email already exists")

        # Create user
        user = User(
            email=data.email,
            hashed_password=hash_password(data.password),
            full_name=data.full_name,
        )
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)

        logger.info("User registered: %s", user.email)

        # Generate tokens
        tokens = self._create_token_pair(user.id)

        return AuthResponse(
            tokens=tokens,
            user=UserResponse.model_validate(user),
        )

    # ── User Login ───────────────────────────────────────────────────────

    async def login(self, email: str, password: str) -> AuthResponse:
        """
        Authenticate user with email and password.

        Args:
            email: User's email address.
            password: Plaintext password to verify.

        Returns:
            AuthResponse with tokens and user profile.

        Raises:
            UnauthorizedError: If credentials are invalid or account is inactive.
        """
        user = await self._get_user_by_email(email)
        if not user:
            raise UnauthorizedError("Invalid email or password")

        if not verify_password(password, user.hashed_password):
            raise UnauthorizedError("Invalid email or password")

        if not user.is_active:
            raise UnauthorizedError("Account is disabled")

        logger.info("User logged in: %s", user.email)

        tokens = self._create_token_pair(user.id)

        return AuthResponse(
            tokens=tokens,
            user=UserResponse.model_validate(user),
        )

    # ── Token Refresh ────────────────────────────────────────────────────

    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """
        Issue a new access token using a valid refresh token.

        Args:
            refresh_token: The refresh JWT to validate.

        Returns:
            New TokenResponse with fresh access and refresh tokens.

        Raises:
            UnauthorizedError: If the refresh token is invalid or expired.
        """
        try:
            payload = decode_token(refresh_token)
        except JWTError:
            raise UnauthorizedError("Invalid or expired refresh token")

        if payload.get("type") != "refresh":
            raise UnauthorizedError("Invalid token type")

        user_id = payload.get("sub")
        if not user_id:
            raise UnauthorizedError("Invalid token payload")

        # Verify user still exists and is active
        user = await self._get_user_by_id(user_id)
        if not user:
            raise UnauthorizedError("User not found")
        if not user.is_active:
            raise UnauthorizedError("Account is disabled")

        logger.info("Token refreshed for user: %s", user.email)

        return self._create_token_pair(user.id)

    # ── Profile Operations ───────────────────────────────────────────────

    async def get_profile(self, user_id: str) -> UserResponse:
        """
        Get user profile by ID.

        Args:
            user_id: The user's UUID.

        Returns:
            UserResponse with the user's public profile.

        Raises:
            NotFoundError: If the user does not exist.
        """
        user = await self._get_user_by_id(user_id)
        if not user:
            raise NotFoundError("User", user_id)
        return UserResponse.model_validate(user)

    async def update_profile(
        self,
        user_id: str,
        data: UpdateProfileRequest,
    ) -> UserResponse:
        """
        Update user profile fields.

        Args:
            user_id: The user's UUID.
            data: Fields to update (full_name, password).

        Returns:
            Updated UserResponse.

        Raises:
            NotFoundError: If the user does not exist.
        """
        user = await self._get_user_by_id(user_id)
        if not user:
            raise NotFoundError("User", user_id)

        if data.full_name is not None:
            user.full_name = data.full_name

        if data.password is not None:
            user.hashed_password = hash_password(data.password)

        await self.db.flush()
        await self.db.refresh(user)

        logger.info("Profile updated for user: %s", user.email)

        return UserResponse.model_validate(user)

    # ── Private Helpers ──────────────────────────────────────────────────

    async def _get_user_by_email(self, email: str) -> User | None:
        """Look up a user by email address."""
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def _get_user_by_id(self, user_id: str) -> User | None:
        """Look up a user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _create_token_pair(user_id: str) -> TokenResponse:
        """Generate a fresh access + refresh token pair."""
        return TokenResponse(
            access_token=create_access_token(subject=user_id),
            refresh_token=create_refresh_token(subject=user_id),
        )
