'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: FastAPI application factory (rpi5.md §13.1).

The module exposes two things:
  - ``create_app()``  — factory used in tests and for custom setup
  - ``app``           — module-level instance for uvicorn

Usage
-----
    # Tests
    from quant_risk.prod.api.app import create_app
    app = create_app()

    # Production (uvicorn)
    uvicorn quant_risk.prod.api.app:app --host 0.0.0.0 --port 8000

    # Development
    uvicorn quant_risk.prod.api.app:app --reload --host 0.0.0.0 --port 8000
'''

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quant_risk.prod.api.config import AppConfig
from quant_risk.prod.api.routers import admin, auth, health, internal, private, public

logger = logging.getLogger(__name__)


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    config:
        Optional AppConfig override.  If None, a fresh AppConfig() is built
        from environment variables (the default production path).

    Returns
    -------
    FastAPI
        Fully configured application instance.
    """
    cfg = config or AppConfig()

    app = FastAPI(
        title="quant-risk API",
        description=(
            "Internal backend for risk.manidmt.es — "
            "volatility regime predictions and portfolio analysis."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # CORS — only needed when the React dev server runs on a different port.
    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Routers
    app.include_router(health.router)
    app.include_router(public.router)
    app.include_router(auth.router)
    app.include_router(private.router)
    app.include_router(admin.router)
    app.include_router(internal.router)

    logger.info("quant-risk API app created.")
    return app


# Module-level instance for uvicorn.
app = create_app()
