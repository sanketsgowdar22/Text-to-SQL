"""
SQLAlchemy ORM models for Authentication & User Management.
"""

from typing import Optional

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.shared.models import TimestampMixin, UUIDMixin


class User(Base, UUIDMixin, TimestampMixin):
    """
    User model representing platform account identity and authorization attributes.

    Inherits from:
    - Base: SQLAlchemy DeclarativeBase
    - UUIDMixin: String UUIDv4 primary key (`id`)
    - TimestampMixin: UTC creation and update timestamps (`created_at`, `updated_at`)
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    full_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id!r}, email={self.email!r}, is_active={self.is_active})>"
