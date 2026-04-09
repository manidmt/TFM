'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: FastAPI dependency injectors (rpi5.md §13.1, §14).

Dependencies
------------
get_auth_db     — yields an open SQLAlchemy Session
get_serving_db  — yields an open ServingDB connection
get_current_user — reads the session cookie and returns the active User
require_admin   — like get_current_user but also checks role == "admin"

Session cookie protocol
-----------------------
- The session token is stored in an HttpOnly cookie named ``qr_session``.
- On each request the session is looked up in Postgres and optionally renewed
  (via maybe_renew_session) if more than RENEWAL_THRESHOLD has elapsed.
- If the session is expired or absent, HTTP 401 is returned.
- must_change_password is NOT enforced here; the /auth/me endpoint exposes the
  flag so the frontend can redirect to the password-change page.
'''

from __future__ import annotations

from typing import Generator

import hmac

import duckdb
from fastapi import Cookie, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from quant_risk.prod.api.config import AppConfig
from quant_risk.prod.auth.db import AuthDB
from quant_risk.prod.auth.models import User
from quant_risk.prod.auth.sessions import get_session_by_token, maybe_renew_session
from quant_risk.prod.serving.duckdb import ServingDB


# ---------------------------------------------------------------------------
# Singleton config and shared database instances (one per process)
# ---------------------------------------------------------------------------

_config = AppConfig()
_auth_db = AuthDB(_config.postgres_url)
_auth_db.create_tables()


def get_config() -> AppConfig:
    return _config


# ---------------------------------------------------------------------------
# Auth DB dependency
# ---------------------------------------------------------------------------

def get_auth_db() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy Session for the auth/portfolio database.

    Uses the process-level shared engine so that connection pools are
    not re-created on every request (fixes C-2).
    """
    with _auth_db.session() as s:
        try:
            yield s
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()


# ---------------------------------------------------------------------------
# Serving DB dependency
# ---------------------------------------------------------------------------

def get_serving_db(
    config: AppConfig = Depends(get_config),
) -> Generator[ServingDB, None, None]:
    """Yield an open ServingDB connection."""
    sdb = ServingDB(config.serving_db_path)
    try:
        yield sdb
    finally:
        sdb.close()


# ---------------------------------------------------------------------------
# Research DB dependency
# ---------------------------------------------------------------------------

def get_research_db(
    config: AppConfig = Depends(get_config),
) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Yield a read-only DuckDB connection to the research database (financial_data.duckdb)."""
    conn = duckdb.connect(config.research_db_path, read_only=True)
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Authentication dependency
# ---------------------------------------------------------------------------

def get_current_user(
    request: Request,
    config: AppConfig = Depends(get_config),
    db: Session = Depends(get_auth_db),
) -> User:
    """Read the session cookie and return the authenticated User.

    Raises HTTP 401 if the cookie is absent, expired, or the user is inactive.
    """
    token: str | None = request.cookies.get(config.cookie_name)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
        )

    sess = get_session_by_token(db, token)
    if sess is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid.",
        )

    user = sess.user
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is inactive.",
        )

    # Transparently renew session when past the renewal threshold.
    # flush() writes the update to the current transaction; the route
    # handler is responsible for the final commit (M-5 fix).
    maybe_renew_session(db, sess)
    db.flush()

    return user


def require_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """Like get_current_user but also enforces role == 'admin'.

    Raises HTTP 403 if the user is not an admin.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required.",
        )
    return current_user


# ---------------------------------------------------------------------------
# Internal bearer-token auth (off-box prediction push)
# ---------------------------------------------------------------------------

def require_internal_token(
    authorization: str | None = Header(default=None),
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
    config: AppConfig = Depends(get_config),
) -> None:
    """Accept a shared bearer token from the off-box prediction producer.

    The token is read from ``QUANT_RISK_INTERNAL_TOKEN`` at application start.
    If the env var is unset or empty the endpoint returns **503** — the RPi5
    intentionally refuses internal writes until the operator opts in by
    configuring the token.

    The header can be either:
      * ``Authorization: Bearer <token>``  (preferred)
      * ``X-Internal-Token: <token>``      (fallback for simpler clients)

    Comparison uses ``hmac.compare_digest`` to avoid timing leaks.
    """
    expected = config.internal_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Internal prediction push is disabled on this server "
                "(QUANT_RISK_INTERNAL_TOKEN not configured)."
            ),
        )

    presented: str | None = None
    if authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            presented = parts[1].strip()
    if presented is None and x_internal_token:
        presented = x_internal_token.strip()

    if not presented or not hmac.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
