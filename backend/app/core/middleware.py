"""
Application middleware.

Provides CORS configuration, request ID tracking, request logging,
and other cross-cutting concerns.
"""

import logging
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.constants import HEADER_REQUEST_ID

logger = logging.getLogger(__name__)


def register_middleware(app: FastAPI) -> None:
    """
    Register all middleware on the FastAPI application.

    Middleware is applied in reverse order (last registered = first executed).
    Order matters: CORS must be outermost, then request ID, then logging.
    """

    # ── Request Logging Middleware ────────────────────────────────────────
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        """Log every request with method, path, status, and duration."""
        start_time = time.perf_counter()
        request_id = request.state.request_id if hasattr(request.state, "request_id") else "N/A"

        logger.info(
            "Request started: %s %s [request_id=%s]",
            request.method,
            request.url.path,
            request_id,
        )

        response: Response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Request completed: %s %s -> %d (%.1fms) [request_id=%s]",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )

        return response

    # ── Request ID Middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        """Assign a unique request ID to every incoming request."""
        request_id = request.headers.get(HEADER_REQUEST_ID, str(uuid.uuid4()))
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers[HEADER_REQUEST_ID] = request_id

        return response

    # ── CORS Middleware ───────────────────────────────────────────────────
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[HEADER_REQUEST_ID],
    )
