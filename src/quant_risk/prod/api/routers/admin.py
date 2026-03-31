'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Admin read-only ops endpoints — requires admin role (rpi5.md §12.3, §17).

Routes
------
GET /api/admin/ops/summary
    Aggregate batch health: asset count, fresh/stale/failed counts.

GET /api/admin/ops/assets
    Latest asset status for every production asset.

GET /api/admin/ops/promotions
    Recent promotion events (latest 20 by default).

Design notes (rpi5.md §12.3)
-----------------------------
- Admin endpoints are READ-ONLY in V1.
- No promotion triggers, no rerun commands, no user management.
- The /ops data is sourced entirely from serving.duckdb via serving/ops.py.
'''

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from quant_risk.prod.api.deps import get_serving_db, require_admin
from quant_risk.prod.auth.models import User
from quant_risk.prod.serving.duckdb import ServingDB
from quant_risk.prod.serving.ops import (
    get_active_bundles,
    get_asset_status_all,
    get_ops_summary,
    get_promotion_history,
    get_recent_runs,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class OpsSummaryOut(BaseModel):
    asset_count: int
    fresh_count: int
    stale_count: int
    failed_count: int
    assets: list[dict]


class AssetStatusOut(BaseModel):
    asset_id: str
    run_status: str | None
    last_forecast_date: str | None
    last_run_at: str | None
    consecutive_failures: int | None


class PromotionEventOut(BaseModel):
    event_id: str
    asset_id: str
    bundle_version: str
    status: str
    promoted_at: str
    previous_version: str | None
    validation_errors: str | None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/ops/summary", response_model=OpsSummaryOut)
def ops_summary(
    serving_db: ServingDB = Depends(get_serving_db),
    _admin: User = Depends(require_admin),
):
    """Aggregate operational health summary."""
    summary = get_ops_summary(serving_db)
    return OpsSummaryOut(**summary)


@router.get("/ops/assets", response_model=list[AssetStatusOut])
def ops_assets(
    serving_db: ServingDB = Depends(get_serving_db),
    _admin: User = Depends(require_admin),
):
    """Latest status for every production asset."""
    rows = get_asset_status_all(serving_db)
    return [
        AssetStatusOut(
            asset_id=r["asset_id"],
            run_status=r.get("run_status"),
            last_forecast_date=str(r["last_forecast_date"]) if r.get("last_forecast_date") else None,
            last_run_at=str(r["last_run_at"]) if r.get("last_run_at") else None,
            consecutive_failures=r.get("consecutive_failures"),
        )
        for r in rows
    ]


@router.get("/ops/promotions", response_model=list[PromotionEventOut])
def ops_promotions(
    limit: int = Query(20, ge=1, le=100),
    asset_id: str | None = Query(None),
    serving_db: ServingDB = Depends(get_serving_db),
    _admin: User = Depends(require_admin),
):
    """Recent promotion/rollback events."""
    rows = get_promotion_history(serving_db, limit=limit, asset_id=asset_id)
    return [
        PromotionEventOut(
            event_id=r["event_id"],
            asset_id=r["asset_id"],
            bundle_version=r["bundle_version"],
            status=r["status"],
            promoted_at=str(r["promoted_at"]),
            previous_version=r.get("previous_version"),
            validation_errors=r.get("validation_errors"),
        )
        for r in rows
    ]
