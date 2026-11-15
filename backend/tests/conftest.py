"""
Shared test fixtures and configuration.

Uses an in-memory SQLite database with StaticPool.
Bypasses the production lifespan to avoid touching the real DB.
"""

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db_session

# Import all ORM models so Base.metadata registers their tables
from app.features.auth.models import User  # noqa: F401

# ── Test Database ────────────────────────────────────────────────────────────

test_engine = create_async_engine(
    "sqlite+aiosqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False,
)

test_session_factory = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@pytest_asyncio.fixture(autouse=True)
async def setup_database() -> AsyncGenerator[None, None]:
    """Create all tables before each test, drop after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a test database session."""
    async with test_session_factory() as session:
        yield session
        await session.rollback()


# ── Test HTTP Client ─────────────────────────────────────────────────────────


async def _override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Override the database session dependency for tests."""
    async with test_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def _create_test_app():
    """Build a test FastAPI app with a no-op lifespan."""
    from contextlib import asynccontextmanager
    from datetime import UTC, datetime

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from app.core.config import get_settings
    from app.core.constants import API_V1_PREFIX
    from app.core.exceptions import register_exception_handlers
    from app.core.middleware import register_middleware
    from app.features.auth.router import router as auth_router
    from app.shared.schemas import HealthResponse, SuccessResponse

    settings = get_settings()

    @asynccontextmanager
    async def noop_lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        yield

    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=noop_lifespan)
    register_middleware(app)
    register_exception_handlers(app)
    app.include_router(auth_router, prefix=f"{API_V1_PREFIX}/auth", tags=["Authentication"])

    @app.get(f"{API_V1_PREFIX}/health", response_model=SuccessResponse[HealthResponse], tags=["System"])
    async def health_check() -> JSONResponse:
        health = HealthResponse(
            status="healthy", version=settings.app_version,
            environment=settings.app_env, timestamp=datetime.now(UTC),
        )
        return JSONResponse(status_code=200, content=SuccessResponse(data=health, message="Service is healthy").model_dump(mode="json"))

    @app.get(f"{API_V1_PREFIX}/health/ready", tags=["System"])
    async def readiness_probe() -> JSONResponse:
        return JSONResponse(status_code=200, content={"success": True, "data": {"ready": True, "checks": {"database": "ok"}}})

    return app


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client with test DB injected."""
    app = _create_test_app()
    app.dependency_overrides[get_db_session] = _override_get_db_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
