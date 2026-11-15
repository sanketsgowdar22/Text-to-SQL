"""
Auth Pydantic schemas.

Request and response models for authentication endpoints.
All schemas use BaseSchema for consistent configuration.
"""

from datetime import datetime

from pydantic import EmailStr, Field

from app.shared.schemas import BaseSchema


# ── Request Schemas ──────────────────────────────────────────────────────────


class RegisterRequest(BaseSchema):
    """Schema for user registration."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password (min 8 characters)",
    )
    full_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="User's full name",
    )


class LoginRequest(BaseSchema):
    """Schema for user login."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class RefreshTokenRequest(BaseSchema):
    """Schema for refreshing an access token."""

    refresh_token: str = Field(..., description="Valid refresh token")


class UpdateProfileRequest(BaseSchema):
    """Schema for updating user profile."""

    full_name: str | None = Field(
        None,
        min_length=1,
        max_length=255,
        description="Updated display name",
    )
    password: str | None = Field(
        None,
        min_length=8,
        max_length=128,
        description="New password (min 8 characters)",
    )


# ── Response Schemas ─────────────────────────────────────────────────────────


class UserResponse(BaseSchema):
    """Public user profile (never includes password)."""

    id: str
    email: str
    full_name: str
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime


class TokenResponse(BaseSchema):
    """JWT token pair returned on login/refresh."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class AuthResponse(BaseSchema):
    """Combined auth response: tokens + user info."""

    tokens: TokenResponse
    user: UserResponse
