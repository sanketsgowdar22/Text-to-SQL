"""
Security dependency guards for FastAPI routes.

Provides injectable dependencies that extract and validate
JWT tokens from the Authorization header, returning the current
user or raising HTTP exceptions.
"""

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.security import decode_token
from app.features.auth.models import User
from app.features.auth.service import AuthService

# Bearer token extractor (auto_error=False so we handle errors ourselves)
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db_session),
) -> User:
    """
    FastAPI dependency: extract and validate the Bearer token,
    return the corresponding active User.

    Usage:
        @router.get("/protected")
        async def protected(user: User = Depends(get_current_user)):
            ...

    Raises:
        UnauthorizedError: If the token is missing, invalid, or the user is inactive.
    """
    if credentials is None:
        raise UnauthorizedError("Missing authorization header")

    token = credentials.credentials
    try:
        payload = decode_token(token)
    except JWTError:
        raise UnauthorizedError("Invalid or expired token")

    if payload.get("type") != "access":
        raise UnauthorizedError("Invalid token type — use an access token")

    user_id = payload.get("sub")
    if not user_id:
        raise UnauthorizedError("Invalid token payload")

    service = AuthService(db)
    user = await service._get_user_by_id(user_id)

    if not user:
        raise UnauthorizedError("User not found")
    if not user.is_active:
        raise UnauthorizedError("Account is disabled")

    return user


async def get_current_superuser(
    user: User = Depends(get_current_user),
) -> User:
    """
    FastAPI dependency: require the current user to be a superuser.

    Usage:
        @router.get("/admin-only")
        async def admin(user: User = Depends(get_current_superuser)):
            ...

    Raises:
        ForbiddenError: If the authenticated user is not a superuser.
    """
    if not user.is_superuser:
        raise ForbiddenError("Superuser access required")
    return user
