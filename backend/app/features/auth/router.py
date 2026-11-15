"""
Auth router — all authentication endpoints.

Endpoints:
    POST /register  — Create a new user account
    POST /login     — Authenticate and receive JWT tokens
    POST /refresh   — Refresh an expired access token
    GET  /me        — Get the current user's profile
    PUT  /me        — Update the current user's profile
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.features.auth.dependencies import get_current_user
from app.features.auth.models import User
from app.features.auth.schemas import (
    AuthResponse,
    LoginRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserResponse,
)
from app.features.auth.service import AuthService
from app.shared.schemas import SuccessResponse

router = APIRouter()


@router.post(
    "/register",
    response_model=SuccessResponse[AuthResponse],
    status_code=201,
    summary="Register a new user",
    description="Create a new user account with email, password, and full name.",
)
async def register(
    data: RegisterRequest,
    db: AsyncSession = Depends(get_db_session),
) -> SuccessResponse[AuthResponse]:
    """Register a new user and return tokens + profile."""
    service = AuthService(db)
    result = await service.register(data)
    return SuccessResponse(data=result, message="User registered successfully")


@router.post(
    "/login",
    response_model=SuccessResponse[AuthResponse],
    summary="Login",
    description="Authenticate with email and password to receive JWT tokens.",
)
async def login(
    data: LoginRequest,
    db: AsyncSession = Depends(get_db_session),
) -> SuccessResponse[AuthResponse]:
    """Authenticate user and return tokens + profile."""
    service = AuthService(db)
    result = await service.login(email=data.email, password=data.password)
    return SuccessResponse(data=result, message="Login successful")


@router.post(
    "/refresh",
    response_model=SuccessResponse[TokenResponse],
    summary="Refresh access token",
    description="Exchange a valid refresh token for a new access/refresh token pair.",
)
async def refresh_token(
    data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db_session),
) -> SuccessResponse[TokenResponse]:
    """Refresh JWT token pair."""
    service = AuthService(db)
    tokens = await service.refresh_token(data.refresh_token)
    return SuccessResponse(data=tokens, message="Token refreshed successfully")


@router.get(
    "/me",
    response_model=SuccessResponse[UserResponse],
    summary="Get current user profile",
    description="Returns the authenticated user's profile information.",
)
async def get_profile(
    user: User = Depends(get_current_user),
) -> SuccessResponse[UserResponse]:
    """Get the current authenticated user's profile."""
    profile = UserResponse.model_validate(user)
    return SuccessResponse(data=profile)


@router.put(
    "/me",
    response_model=SuccessResponse[UserResponse],
    summary="Update current user profile",
    description="Update the authenticated user's name or password.",
)
async def update_profile(
    data: UpdateProfileRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> SuccessResponse[UserResponse]:
    """Update the current user's profile."""
    service = AuthService(db)
    updated = await service.update_profile(user_id=user.id, data=data)
    return SuccessResponse(data=updated, message="Profile updated successfully")
