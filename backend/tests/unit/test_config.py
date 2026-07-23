"""
Tests for the configuration module.

Validates that settings load correctly from environment variables
and defaults are applied when no overrides are present.
"""

import pytest

from app.core.config import Settings


@pytest.mark.unit
class TestSettings:
    """Tests for the Settings configuration class."""

    def test_default_app_name(self) -> None:
        """Default app name should be ADIP."""
        settings = Settings()
        assert settings.app_name == "ADIP"

    def test_default_version(self) -> None:
        """Default version should be 0.1.0."""
        settings = Settings()
        assert settings.app_version == "0.1.0"

    def test_default_environment(self) -> None:
        """Default environment should be development."""
        settings = Settings()
        assert settings.app_env == "development"

    def test_is_development(self) -> None:
        """is_development should return True for development env."""
        settings = Settings()
        assert settings.is_development is True
        assert settings.is_production is False

    def test_is_production(self) -> None:
        """is_production should return True for production env."""
        settings = Settings(app_env="production")
        assert settings.is_production is True
        assert settings.is_development is False

    def test_default_database_url(self) -> None:
        """Default database URL should use SQLite."""
        settings = Settings()
        assert "sqlite" in settings.database_url

    def test_database_path_for_sqlite(self) -> None:
        """database_path should return a Path for SQLite URLs."""
        settings = Settings(database_url="sqlite+aiosqlite:///./test.db")
        assert settings.database_path is not None
        assert str(settings.database_path).endswith("test.db")

    def test_database_path_for_non_sqlite(self) -> None:
        """database_path should return None for non-SQLite URLs."""
        settings = Settings(database_url="postgresql+asyncpg://user:pass@localhost/db")
        assert settings.database_path is None

    def test_cors_origins_from_string(self) -> None:
        """CORS origins should parse from JSON string."""
        settings = Settings(cors_origins='["http://localhost:3000"]')
        assert settings.cors_origins == ["http://localhost:3000"]

    def test_cors_origins_from_csv(self) -> None:
        """CORS origins should parse from comma-separated string."""
        settings = Settings(cors_origins="http://localhost:3000,http://localhost:8080")
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://localhost:8080" in settings.cors_origins

    def test_cors_origins_from_list(self) -> None:
        """CORS origins should accept a list directly."""
        origins = ["http://localhost:3000", "http://localhost:8080"]
        settings = Settings(cors_origins=origins)
        assert settings.cors_origins == origins

    def test_default_secret_key(self) -> None:
        """Default secret key should be the placeholder."""
        settings = Settings()
        assert "CHANGE-THIS" in settings.secret_key

    def test_default_token_expiry(self) -> None:
        """Default token expiry should be 30 minutes access, 7 days refresh."""
        settings = Settings()
        assert settings.access_token_expire_minutes == 30
        assert settings.refresh_token_expire_days == 7
