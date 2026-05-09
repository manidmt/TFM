'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Health check endpoints (rpi5.md §20).

Routes
------
GET /health          — liveness probe (always 200 if the process is alive)
GET /health/ready    — readiness probe (checks DB connectivity)
'''

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from quant_risk.prod.api.config import AppConfig
from quant_risk.prod.api.deps import _auth_db, get_config
from quant_risk.prod.serving.duckdb import ServingDB

router = APIRouter(tags=["health"])


@router.get("/health", status_code=status.HTTP_200_OK)
def liveness():
    """Liveness probe — returns 200 if the API process is running."""
    return {"status": "ok"}


@router.get("/health/ready")
def readiness(config: AppConfig = Depends(get_config)):
    """Readiness probe — checks that both databases are reachable.

    Returns 200 with per-component status, or 503 if any component is down.
    """
    checks: dict[str, bool] = {}

    # Auth DB (Postgres / SQLite) — reuse the shared engine, no new pool.
    try:
        checks["auth_db"] = _auth_db.ping()
    except Exception:
        checks["auth_db"] = False

    # Serving DB (DuckDB)
    try:
        sdb = ServingDB(config.serving_db_path)
        checks["serving_db"] = sdb.ping()
        sdb.close()
    except Exception:
        checks["serving_db"] = False

    all_ok = all(checks.values())
    http_status = status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=http_status,
        content={"status": "ok" if all_ok else "degraded", "checks": checks},
    )
