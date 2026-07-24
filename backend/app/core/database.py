"""
Database engine and session management.

Provides async SQLAlchemy engine, session factory, and dependency injection
for FastAPI route handlers.
"""

from collections.abc import AsyncGenerator

from sqlalchemy import MetaData, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings
from app.core.constants import DB_NAMING_CONVENTION

settings = get_settings()

# ── Engine Configuration ─────────────────────────────────────────────────────

# SQLite-specific: enable WAL mode and foreign keys
_engine_kwargs: dict[str, object] = {
    "echo": settings.debug,
    "pool_pre_ping": True,
}

# SQLite requires special handling for async
if "sqlite" in settings.database_url:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_async_engine(
    settings.database_url,
    **_engine_kwargs,
)


# Enable SQLite foreign key enforcement (must be done per connection)
@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]
    """Enable foreign keys and WAL mode for SQLite connections."""
    if "sqlite" in settings.database_url:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


# ── Session Factory ──────────────────────────────────────────────────────────

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# ── Base Model ───────────────────────────────────────────────────────────────

metadata = MetaData(naming_convention=DB_NAMING_CONVENTION)


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.

    Uses a consistent naming convention for indexes, constraints,
    and foreign keys to ensure predictable migration behavior.
    """

    metadata = metadata


# ── Dependency Injection ─────────────────────────────────────────────────────


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.

    Yields an async session and ensures it is closed after the request
    completes, regardless of success or failure.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Lifecycle Management ─────────────────────────────────────────────────────


async def init_db() -> None:
    """
    Initialize the database by creating all tables.

    This is primarily used for development and testing.
    In production, use Alembic migrations instead.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose of the database engine and close all connections."""
    await engine.dispose()
