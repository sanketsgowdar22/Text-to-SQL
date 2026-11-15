"""
User ORM model.

Defines the User table with email, hashed password, profile fields,
and role-based flags. Uses shared mixins for UUID primary key and timestamps.
"""

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.shared.models import TimestampMixin, UUIDMixin


class User(Base, UUIDMixin, TimestampMixin):
    """
    User account model.

    Attributes:
        id: UUID primary key (from UUIDMixin).
        email: Unique email address used for authentication.
        hashed_password: Bcrypt-hashed password.
        full_name: User's display name.
        is_active: Whether the account is enabled.
        is_superuser: Whether the user has admin privileges.
        created_at: Account creation timestamp (from TimestampMixin).
        updated_at: Last modification timestamp (from TimestampMixin).
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
    full_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default="1",
        nullable=False,
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="0",
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id!r}, email={self.email!r})>"
