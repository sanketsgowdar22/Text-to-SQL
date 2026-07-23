"""
Shared Pydantic schemas for consistent API responses.

Provides base response wrappers used across all features
to ensure a uniform API response structure.
"""

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class BaseSchema(BaseModel):
    """Base schema with common Pydantic configuration."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class SuccessResponse(BaseModel, Generic[T]):
    """
    Standard success response wrapper.

    Ensures all successful API responses follow the same structure:
    {
        "success": true,
        "data": <payload>,
        "message": "optional message"
    }
    """

    success: bool = True
    data: T
    message: str | None = None


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Paginated response wrapper for list endpoints.

    {
        "success": true,
        "data": [...],
        "pagination": {
            "page": 1,
            "page_size": 20,
            "total_items": 100,
            "total_pages": 5
        }
    }
    """

    success: bool = True
    data: list[T]
    pagination: "PaginationMeta"


class PaginationMeta(BaseModel):
    """Pagination metadata included in paginated responses."""

    page: int = Field(ge=1, description="Current page number")
    page_size: int = Field(ge=1, description="Items per page")
    total_items: int = Field(ge=0, description="Total number of items")
    total_pages: int = Field(ge=0, description="Total number of pages")


class ErrorDetail(BaseModel):
    """Structured error detail for API error responses."""

    type: str
    message: str
    detail: Any = None


class ErrorResponse(BaseModel):
    """
    Standard error response wrapper.

    {
        "success": false,
        "error": {
            "type": "NotFoundError",
            "message": "Resource not found",
            "detail": null
        }
    }
    """

    success: bool = False
    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check endpoint response."""

    status: str = "healthy"
    version: str
    environment: str
    timestamp: datetime
