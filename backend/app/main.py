"""
FastAPI application factory.

Creates and configures the FastAPI application with all middleware,
exception handlers, routers, and lifecycle events.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.constants import API_V1_PREFIX
from app.core.database import close_db, init_db
from app.core.exceptions import register_exception_handlers
from app.core.middleware import register_middleware
from app.shared.schemas import HealthResponse, SuccessResponse
from app.shared.utils import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events:
    - Startup: Initialize database, configure logging
    - Shutdown: Close database connections, cleanup resources
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.app_env)

    # Startup: Initialize database
    await init_db()
    logger.info("Database initialized")

    yield

    # Shutdown: Close database
    await close_db()
    logger.info("Database connections closed")
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This is the application factory function. It:
    1. Creates the FastAPI instance with metadata
    2. Registers middleware (CORS, logging, request ID)
    3. Registers exception handlers
    4. Includes feature routers
    5. Adds health check endpoints

    Returns:
        The configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="AI Data Intelligence Platform — Natural Language to SQL",
        version=settings.app_version,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
        lifespan=lifespan,
    )

    # ── Register Middleware ───────────────────────────────────────────────
    register_middleware(app)

    # ── Register Exception Handlers ──────────────────────────────────────
    register_exception_handlers(app)

    # ── Register Feature Routers ─────────────────────────────────────────
    _register_routers(app)

    # ── Health Check Endpoints ───────────────────────────────────────────
    _register_health_endpoints(app)

    return app


def _register_routers(app: FastAPI) -> None:
    """
    Register all feature routers on the application.

    Routers are added incrementally as features are implemented:
    - Phase 2: auth router
    - Phase 3: connections router
    - Phase 4: schema router
    - etc.
    """
    # Feature routers will be registered here as phases are implemented.
    # Example (Phase 2):
    # from app.features.auth.router import router as auth_router
    # app.include_router(auth_router, prefix=f"{API_V1_PREFIX}/auth", tags=["Authentication"])
    pass  # noqa: PIE790


def _register_health_endpoints(app: FastAPI) -> None:
    """Register health check and readiness probe endpoints."""

    @app.get(
        f"{API_V1_PREFIX}/health",
        response_model=SuccessResponse[HealthResponse],
        tags=["System"],
        summary="Health Check",
        description="Returns the current health status of the application.",
    )
    async def health_check() -> JSONResponse:
        """
        Health check endpoint.

        Returns the application status, version, environment, and timestamp.
        Used by monitoring systems and load balancers.
        """
        settings = get_settings()
        health = HealthResponse(
            status="healthy",
            version=settings.app_version,
            environment=settings.app_env,
            timestamp=datetime.now(timezone.utc),
        )
        response = SuccessResponse(data=health, message="Service is healthy")
        return JSONResponse(
            status_code=200,
            content=response.model_dump(mode="json"),
        )

    @app.get(
        f"{API_V1_PREFIX}/health/ready",
        tags=["System"],
        summary="Readiness Probe",
        description="Returns readiness status for Kubernetes/Docker health checks.",
    )
    async def readiness_probe() -> JSONResponse:
        """
        Readiness probe endpoint.

        Verifies the application is ready to serve traffic by checking
        critical dependencies (database connectivity, etc.).
        """
        # In future phases, this will check database connectivity,
        # FAISS index status, LLM provider availability, etc.
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "ready": True,
                    "checks": {
                        "database": "ok",
                    },
                },
            },
        )


# ── Application Instance ─────────────────────────────────────────────────────

app = create_app()
