"""
Pagination helpers.

Provides utilities for cursor-based and offset-based pagination
used across all list endpoints.
"""

import math

from fastapi import Query

from app.core.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, MIN_PAGE_SIZE
from app.shared.schemas import PaginationMeta


class PaginationParams:
    """
    FastAPI dependency for extracting pagination query parameters.

    Usage:
        @router.get("/items")
        async def list_items(pagination: PaginationParams = Depends()):
            ...
    """

    def __init__(
        self,
        page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
        page_size: int = Query(
            default=DEFAULT_PAGE_SIZE,
            ge=MIN_PAGE_SIZE,
            le=MAX_PAGE_SIZE,
            description="Number of items per page",
        ),
    ) -> None:
        self.page = page
        self.page_size = page_size

    @property
    def offset(self) -> int:
        """Calculate the SQL OFFSET value."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Return the SQL LIMIT value (same as page_size)."""
        return self.page_size


def build_pagination_meta(
    page: int,
    page_size: int,
    total_items: int,
) -> PaginationMeta:
    """
    Build pagination metadata from current page info and total count.

    Args:
        page: Current page number (1-indexed).
        page_size: Number of items per page.
        total_items: Total number of items across all pages.

    Returns:
        PaginationMeta with calculated total_pages.
    """
    total_pages = math.ceil(total_items / page_size) if page_size > 0 else 0
    return PaginationMeta(
        page=page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
    )
