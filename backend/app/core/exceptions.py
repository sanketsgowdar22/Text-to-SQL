"""
Custom exception classes and FastAPI exception handlers.

Provides a consistent error response format across the entire API.
All custom exceptions inherit from ADIPBaseError to ensure
uniform error handling and response structure.
"""

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# ── Base Exception ───────────────────────────────────────────────────────────


class ADIPBaseError(Exception):
    """
    Base exception for all ADIP application errors.

    Provides a consistent structure with status code, error message,
    and optional detail payload for all custom exceptions.
    """

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: Any = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)


# ── Specific Exceptions ─────────────────────────────────────────────────────


class NotFoundError(ADIPBaseError):
    """Raised when a requested resource is not found."""

    def __init__(self, resource: str = "Resource", resource_id: Any = None) -> None:
        detail = {"resource": resource}
        if resource_id is not None:
            detail["id"] = str(resource_id)
        super().__init__(
            message=f"{resource} not found",
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )


class ConflictError(ADIPBaseError):
    """Raised when a resource already exists or conflicts with current state."""

    def __init__(self, message: str = "Resource already exists") -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
        )


class UnauthorizedError(ADIPBaseError):
    """Raised when authentication fails or is missing."""

    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class ForbiddenError(ADIPBaseError):
    """Raised when the user lacks permission to access a resource."""

    def __init__(self, message: str = "Insufficient permissions") -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
        )


class BadRequestError(ADIPBaseError):
    """Raised when the request is malformed or contains invalid data."""

    def __init__(self, message: str = "Bad request", detail: Any = None) -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )


class DatabaseConnectionError(ADIPBaseError):
    """Raised when a database connection attempt fails."""

    def __init__(self, message: str = "Database connection failed", detail: Any = None) -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )


class QueryExecutionError(ADIPBaseError):
    """Raised when a SQL query execution fails."""

    def __init__(self, message: str = "Query execution failed", detail: Any = None) -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
        )


class LLMProviderError(ADIPBaseError):
    """Raised when an LLM provider request fails."""

    def __init__(self, message: str = "LLM provider error", detail: Any = None) -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        )


class RateLimitError(ADIPBaseError):
    """Raised when a rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )


# ── Error Response Schema ────────────────────────────────────────────────────


def _build_error_response(
    status_code: int,
    message: str,
    error_type: str,
    detail: Any = None,
) -> JSONResponse:
    """Build a consistent JSON error response."""
    content: dict[str, Any] = {
        "success": False,
        "error": {
            "type": error_type,
            "message": message,
        },
    }
    if detail is not None:
        content["error"]["detail"] = detail
    return JSONResponse(status_code=status_code, content=content)


# ── Exception Handlers ──────────────────────────────────────────────────────


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all custom exception handlers on the FastAPI application.

    This ensures every error — custom, validation, or unexpected —
    returns a consistent JSON structure.
    """

    @app.exception_handler(ADIPBaseError)
    async def adip_exception_handler(
        request: Request,
        exc: ADIPBaseError,
    ) -> JSONResponse:
        """Handle all ADIP custom exceptions."""
        return _build_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_type=type(exc).__name__,
            detail=exc.detail,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Handle Pydantic validation errors with a consistent format."""
        errors = []
        for error in exc.errors():
            field = " -> ".join(str(loc) for loc in error["loc"]) if error["loc"] else "unknown"
            errors.append(
                {
                    "field": field,
                    "message": error["msg"],
                    "type": error["type"],
                }
            )
        return _build_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Validation error",
            error_type="ValidationError",
            detail=errors,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle any unhandled exceptions to prevent stack trace leaks."""
        return _build_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Internal server error",
            error_type="InternalError",
        )
