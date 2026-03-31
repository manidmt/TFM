'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: AuthDB — SQLAlchemy engine + session factory wrapper (rpi5.md §11.3).

Usage
-----
    db = AuthDB()                        # reads QUANT_RISK_POSTGRES_URL env var
    db = AuthDB("postgresql://...")      # explicit URL
    db = AuthDB("sqlite:///:memory:")    # in-process SQLite (tests / dev)

    db.create_tables()                   # idempotent DDL
    with db.session() as s:
        user = users.get_user_by_email(s, "admin@example.com")

The URL falls back to SQLite in-memory when the environment variable is absent.
SQLite is sufficient for development and CI; Postgres is used in production.
'''

from __future__ import annotations

import logging
import os

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from quant_risk.prod.auth.models import Base

logger = logging.getLogger(__name__)

_ENV_VAR = "QUANT_RISK_POSTGRES_URL"
_SQLITE_FALLBACK = "sqlite:///:memory:"


class AuthDB:
    """Thin wrapper around a SQLAlchemy engine for the auth/session schema.

    Parameters
    ----------
    url:
        SQLAlchemy database URL.  If None, reads ``QUANT_RISK_POSTGRES_URL``
        from the environment; falls back to SQLite in-memory.
    echo:
        Pass True to enable SQLAlchemy query logging (useful for debugging).
    """

    def __init__(self, url: str | None = None, *, echo: bool = False) -> None:
        resolved = url or os.environ.get(_ENV_VAR) or _SQLITE_FALLBACK
        self._url = resolved

        # SQLite in-memory: use StaticPool so all sessions share one connection
        # and see the same tables created by create_tables().
        if resolved == _SQLITE_FALLBACK:
            self._engine = create_engine(
                resolved,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=echo,
            )
        else:
            self._engine = create_engine(resolved, echo=echo)

        # Enable foreign-key enforcement for SQLite (disabled by default).
        if resolved.startswith("sqlite"):
            @event.listens_for(self._engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, _connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        self._factory: sessionmaker[Session] = sessionmaker(
            bind=self._engine, expire_on_commit=False
        )
        logger.debug("AuthDB initialised — url=%s", self._url.split("@")[-1])

    # ------------------------------------------------------------------
    # DDL
    # ------------------------------------------------------------------

    def create_tables(self) -> None:
        """Create all auth tables (idempotent — uses CREATE TABLE IF NOT EXISTS)."""
        Base.metadata.create_all(self._engine)
        logger.info("AuthDB tables created/verified.")

    def drop_tables(self) -> None:
        """Drop all auth tables.  **Destructive** — for tests and migrations only."""
        Base.metadata.drop_all(self._engine)
        logger.warning("AuthDB tables dropped.")

    # ------------------------------------------------------------------
    # Session factory
    # ------------------------------------------------------------------

    def session(self) -> Session:
        """Return a new SQLAlchemy Session (caller manages commit/close).

        Prefer using as a context manager::

            with db.session() as s:
                ...
        """
        return self._factory()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Dispose the connection pool."""
        self._engine.dispose()

    @property
    def url(self) -> str:
        """Return the database URL (with password redacted for logging)."""
        return self._url.split("@")[-1] if "@" in self._url else self._url

    def ping(self) -> bool:
        """Return True if the database is reachable."""
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
