"""
Application configuration management.

Uses Pydantic Settings to load configuration from environment variables
and .env files with type validation and default values.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables or a .env file.
    The .env file is loaded automatically from the backend root directory.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "ADIP"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./adip.db"

    # ── Security ─────────────────────────────────────────────────────────────
    secret_key: str = "CHANGE-THIS-TO-A-RANDOM-SECRET-KEY-IN-PRODUCTION"  # noqa: S105
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    algorithm: str = "HS256"

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # ── Gemini API (Phase 7+) ────────────────────────────────────────────────
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"

    # ── Embedding Model (Phase 5+) ──────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Derived Properties ───────────────────────────────────────────────────

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env == "production"

    @property
    def database_path(self) -> Path | None:
        """Extract SQLite database file path, if applicable."""
        if "sqlite" in self.database_url:
            # Extract path after sqlite+aiosqlite:///
            db_path = self.database_url.split("///")[-1]
            return Path(db_path)
        return None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            import json

            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(origin) for origin in parsed]
            except (json.JSONDecodeError, TypeError):
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        if isinstance(v, list):
            return [str(origin) for origin in v]
        return v  # type: ignore[no-any-return]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses lru_cache to ensure settings are loaded only once
    and reused across the application lifecycle.

    Returns:
        Settings: The application configuration instance.
    """
    return Settings()
