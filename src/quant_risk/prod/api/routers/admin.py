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

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from quant_risk.prod.api.deps import get_auth_db, get_serving_db, require_admin
from quant_risk.prod.auth.models import User
from quant_risk.prod.auth.users import approve_user, get_user_by_id, reject_user, set_active
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

class UserAdminOut(BaseModel):
    user_id: str
    email: str
    role: str
    is_active: bool
    is_approved: bool
    created_at: str


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


# ---------------------------------------------------------------------------
# User management routes
# ---------------------------------------------------------------------------

def _user_admin_out(u: User) -> UserAdminOut:
    return UserAdminOut(
        user_id=u.id,
        email=u.email,
        role=u.role,
        is_active=u.is_active,
        is_approved=u.is_approved,
        created_at=u.created_at.isoformat(),
    )


@router.get("/users", response_model=list[UserAdminOut])
def list_users(
    db: Session = Depends(get_auth_db),
    _admin: User = Depends(require_admin),
):
    """List all user accounts."""
    users = db.execute(select(User).order_by(User.created_at.desc())).scalars().all()
    return [_user_admin_out(u) for u in users]


@router.get("/users/pending", response_model=list[UserAdminOut])
def list_pending_users(
    db: Session = Depends(get_auth_db),
    _admin: User = Depends(require_admin),
):
    """List users awaiting approval."""
    users = db.execute(
        select(User).where(User.is_approved == False).order_by(User.created_at.desc())  # noqa: E712
    ).scalars().all()
    return [_user_admin_out(u) for u in users]


@router.post("/users/{user_id}/approve", response_model=UserAdminOut)
def approve_user_endpoint(
    user_id: str,
    db: Session = Depends(get_auth_db),
    _admin: User = Depends(require_admin),
):
    """Approve a pending user account."""
    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    approve_user(db, user)
    db.commit()
    return _user_admin_out(user)


@router.post("/users/{user_id}/reject", response_model=UserAdminOut)
def reject_user_endpoint(
    user_id: str,
    db: Session = Depends(get_auth_db),
    _admin: User = Depends(require_admin),
):
    """Reject a pending user account."""
    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    reject_user(db, user)
    db.commit()
    return _user_admin_out(user)


@router.post("/users/{user_id}/deactivate", response_model=UserAdminOut)
def deactivate_user_endpoint(
    user_id: str,
    db: Session = Depends(get_auth_db),
    _admin: User = Depends(require_admin),
):
    """Deactivate an approved user account."""
    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    set_active(db, user, False)
    db.commit()
    return _user_admin_out(user)


@router.post("/users/{user_id}/activate", response_model=UserAdminOut)
def activate_user_endpoint(
    user_id: str,
    db: Session = Depends(get_auth_db),
    _admin: User = Depends(require_admin),
):
    """Re-activate a deactivated user account."""
    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    set_active(db, user, True)
    db.commit()
    return _user_admin_out(user)


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
