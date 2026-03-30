'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Application configuration for the FastAPI backend (rpi5.md §13.1).

All settings are read from environment variables with sensible defaults.
Production values are set via Docker Compose environment or systemd drop-ins.

Environment variables
---------------------
QUANT_RISK_POSTGRES_URL    — SQLAlchemy URL for AuthDB (users, sessions, portfolios)
QUANT_RISK_SERVING_DB_PATH — Path to serving.duckdb
QUANT_RISK_COOKIE_SECRET   — Secret for cookie signing (generate with secrets.token_hex(32))
QUANT_RISK_CORS_ORIGINS    — Comma-separated list of allowed CORS origins
QUANT_RISK_COOKIE_SECURE   — Set to "false" in local dev (default: true)
'''

from __future__ import annotations

import os


class AppConfig:
    """Runtime configuration derived from environment variables.

    Attributes
    ----------
    postgres_url:
        SQLAlchemy DB URL for the auth/portfolio Postgres database.
    serving_db_path:
        Filesystem path to ``serving.duckdb`` (or ``:memory:`` for tests).
    cookie_secret:
        Secret used to sign session cookies.
    cookie_secure:
        Whether to set the Secure flag on session cookies.
        Should be True in production (HTTPS only).
    cookie_httponly:
        Always True — session cookies are not accessible from JavaScript.
    cookie_samesite:
        SameSite policy.  ``lax`` is the recommended default.
    cookie_name:
        Name of the session cookie.
    cors_origins:
        List of allowed CORS origins.
    """

    def __init__(self) -> None:
        self.postgres_url: str | None = os.environ.get("QUANT_RISK_POSTGRES_URL")
        self.serving_db_path: str | None = os.environ.get("QUANT_RISK_SERVING_DB_PATH")
        self.research_db_path: str = os.environ.get(
            "QUANT_RISK_DB_PATH", "data/db/financial_data.duckdb"
        )
        self.cookie_secret: str = os.environ.get(
            "QUANT_RISK_COOKIE_SECRET", "change-me-in-production"
        )
        self.cookie_secure: bool = (
            os.environ.get("QUANT_RISK_COOKIE_SECURE", "true").lower() != "false"
        )
        self.cookie_httponly: bool = True
        self.cookie_samesite: str = "lax"
        self.cookie_name: str = "qr_session"
        self.cors_origins: list[str] = [
            o.strip()
            for o in os.environ.get("QUANT_RISK_CORS_ORIGINS", "").split(",")
            if o.strip()
        ]
