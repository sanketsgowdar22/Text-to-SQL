"""
Unit tests for User SQLAlchemy ORM model (Task 2.1).
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.features.auth.models import User


@pytest.mark.asyncio
class TestUserModel:
    """Test suite for User model database operations and default attributes."""

    async def test_user_persistence_and_defaults(self, db_session: AsyncSession) -> None:
        user = User(
            email="testuser@example.com",
            hashed_password="hashed_secret_password",
            full_name="Test User",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        assert user.id is not None
        assert len(user.id) == 36  # UUID string length
        assert user.email == "testuser@example.com"
        assert user.hashed_password == "hashed_secret_password"
        assert user.full_name == "Test User"
        assert user.is_active is True
        assert user.is_superuser is False
        assert user.created_at is not None
        assert user.updated_at is not None
        assert f"<User(id={user.id!r}" in repr(user)

    async def test_user_query_by_email(self, db_session: AsyncSession) -> None:
        user = User(email="unique@example.com", hashed_password="pwd")
        db_session.add(user)
        await db_session.commit()

        result = await db_session.execute(select(User).where(User.email == "unique@example.com"))
        fetched_user = result.scalar_one_or_none()

        assert fetched_user is not None
        assert fetched_user.id == user.id

    async def test_superuser_creation(self, db_session: AsyncSession) -> None:
        admin = User(
            email="admin@example.com",
            hashed_password="pwd",
            is_superuser=True,
        )
        db_session.add(admin)
        await db_session.commit()
        await db_session.refresh(admin)

        assert admin.is_superuser is True
