'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: SQLAlchemy ORM models for the auth/session/portfolio layer
              (rpi5.md §11.3, §14, §15).

Tables
------
users
    email + scrypt-hash authentication, role enum, must_change_password flag.

sessions
    Server-side session tokens persisted in Postgres.
    Each row is one live session; expired or invalidated sessions are deleted.

portfolios
    One row per named portfolio owned by a user (rpi5.md §15.2).

portfolio_positions
    One row per position within a portfolio.
    Positions are replaced atomically — no history stored in V1.

Design notes
------------
- Primary keys use UUID stored as String(36) for SQLite and Postgres compat in tests.
- DateTime(timezone=True) stores UTC timestamps; both SQLite and Postgres accept this.
- `AppSession` avoids collision with SQLAlchemy's own `Session` class.
- Cascade delete: user → portfolios → positions; user → sessions.
'''

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# Valid role values — enforced at application layer, not via DB CHECK for SQLite compat.
VALID_ROLES = frozenset({"user", "admin"})


class User(Base):
    """Production user account.

    Attributes
    ----------
    id:
        UUID primary key (stored as string for cross-DB compat).
    email:
        Unique login identifier, case-insensitive (lowercased on write).
    password_hash:
        scrypt-derived hash; never the plaintext password.
    role:
        ``'user'`` or ``'admin'``.
    must_change_password:
        True for newly created users; cleared after first successful password change.
    is_active:
        Soft-delete flag; inactive users cannot authenticate.
    created_at / updated_at:
        UTC timestamps.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="user")
    must_change_password: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_approved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    sessions: Mapped[list[AppSession]] = relationship(
        "AppSession", back_populates="user", cascade="all, delete-orphan"
    )
    portfolios: Mapped[list[Portfolio]] = relationship(
        "Portfolio", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"User(email={self.email!r}, role={self.role!r}, active={self.is_active})"


class AppSession(Base):
    """Server-side session record.

    One row = one active browser/client session.
    Rows are deleted on logout or expiry.

    Attributes
    ----------
    id:
        UUID primary key.
    user_id:
        FK → users.id (CASCADE DELETE).
    token:
        Opaque URL-safe token issued to the client (stored verbatim for V1;
        a production hardening step would store only the hash).
    created_at:
        When the session was first created.
    expires_at:
        Hard expiry; sessions past this timestamp are treated as invalid.
    renewed_at:
        Last renewal timestamp; updated on each active request.
    """

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    renewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    user: Mapped[User] = relationship("User", back_populates="sessions")

    __table_args__ = (
        Index("ix_sessions_token", "token"),
        Index("ix_sessions_user_id", "user_id"),
    )

    def __repr__(self) -> str:
        return (
            f"AppSession(user_id={self.user_id!r}, expires_at={self.expires_at!r})"
        )


# ---------------------------------------------------------------------------
# Portfolio models (rpi5.md §15)
# ---------------------------------------------------------------------------

class Portfolio(Base):
    """A named portfolio owned by one user.

    Each user may have multiple portfolios.  V1 stores only the current
    state — no detailed version history.

    Attributes
    ----------
    id:
        UUID primary key.
    user_id:
        FK → users.id (CASCADE DELETE).
    name:
        User-supplied portfolio name (must be unique per user).
    created_at / updated_at:
        UTC timestamps.
    positions:
        Ordered list of PortfolioPosition rows.
    """

    __tablename__ = "portfolios"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_analysis_signal: Mapped[str | None] = mapped_column(
        String(20), nullable=True, default=None
    )
    last_analysis_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    positions: Mapped[list[PortfolioPosition]] = relationship(
        "PortfolioPosition",
        back_populates="portfolio",
        cascade="all, delete-orphan",
        order_by="PortfolioPosition.label",
    )
    user: Mapped[User] = relationship("User", back_populates="portfolios")

    __table_args__ = (
        Index("ix_portfolios_user_id", "user_id"),
        UniqueConstraint("user_id", "name", name="uq_portfolios_user_name"),
    )

    def __repr__(self) -> str:
        return f"Portfolio(name={self.name!r}, user_id={self.user_id!r})"


class PortfolioPosition(Base):
    """One position (row) in a portfolio.

    A position maps a freeform label (e.g. ``AAPL``) to one of the six
    production ``asset_id`` proxies (e.g. ``us_equities``).

    Attributes
    ----------
    id:
        UUID primary key.
    portfolio_id:
        FK → portfolios.id (CASCADE DELETE).
    label:
        Freeform ticker/label supplied by the user (e.g. ``AAPL``, ``BBVA``).
    weight_pct:
        Weight in percent (0–100).  Validation and normalisation are done
        in the analysis service, not enforced at DB level for flexibility.
    proxy_asset_id:
        One of the six production asset IDs (e.g. ``us_equities``).
    created_at:
        UTC timestamp of last replacement.
    """

    __tablename__ = "portfolio_positions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    portfolio_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
    )
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    weight_pct: Mapped[float] = mapped_column(Float, nullable=False)
    proxy_asset_id: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    portfolio: Mapped[Portfolio] = relationship(
        "Portfolio", back_populates="positions"
    )

    __table_args__ = (
        Index("ix_portfolio_positions_portfolio_id", "portfolio_id"),
    )

    def __repr__(self) -> str:
        return (
            f"PortfolioPosition(label={self.label!r}, "
            f"weight_pct={self.weight_pct}, proxy={self.proxy_asset_id!r})"
        )
