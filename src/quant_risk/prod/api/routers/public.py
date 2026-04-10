'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Public prediction endpoints — no authentication required (rpi5.md §12.1, §17).

Routes
------
GET /api/public/assets
    List all production asset IDs and their display names.

GET /api/public/predictions/latest
    Latest prediction for every production asset.
    Returns: asset_id, predicted_class, p_low, p_medium, p_high, forecast_date.
    Deliberately omits: bundle_version, model_type, operational flags.

GET /api/public/predictions/history?asset_id=us_equities&limit=90
    Historical predictions for one asset (newest first).
    Returns same fields as /latest.

Design notes (rpi5.md §12.1)
-----------------------------
- Public pages must NOT expose bundle_version, model_type, or stale/failure flags.
- The primary user-facing output is predicted_class + class probabilities.
'''

from __future__ import annotations

from datetime import date, timedelta

import duckdb
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from quant_risk.prod.api.deps import get_research_db, get_serving_db
from quant_risk.prod.assets import load_asset_catalog
from quant_risk.prod.serving.duckdb import ServingDB
from quant_risk.prod.serving.predictions import (
    get_all_latest_predictions,
    get_latest_prediction,
    get_prediction_history,
)

router = APIRouter(prefix="/api/public", tags=["public"])

# Default assets config path — override via env or test injection if needed.
_ASSETS_CONFIG = "config/prod/assets.yaml"


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class AssetOut(BaseModel):
    asset_id: str
    source_ticker: str
    display_name: str


class PredictionOut(BaseModel):
    asset_id: str
    forecast_date: str
    predicted_class: str
    p_low: float
    p_medium: float
    p_high: float


class PricePoint(BaseModel):
    date: str
    close: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_prediction_out(row: dict) -> PredictionOut:
    return PredictionOut(
        asset_id=row["asset_id"],
        forecast_date=str(row["forecast_date"]),
        predicted_class=row["predicted_class"],
        p_low=row["p_low"],
        p_medium=row["p_medium"],
        p_high=row["p_high"],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/assets", response_model=list[AssetOut])
def list_assets():
    """Return all production asset IDs with display names."""
    try:
        catalog = load_asset_catalog(_ASSETS_CONFIG)
    except FileNotFoundError:
        return []
    return [
        AssetOut(
            asset_id=a.asset_id,
            source_ticker=a.source_ticker,
            display_name=getattr(a, "display_name", a.asset_id),
        )
        for a in catalog
    ]


@router.get("/predictions/latest", response_model=list[PredictionOut])
def predictions_latest(serving_db: ServingDB = Depends(get_serving_db)):
    """Return the latest prediction for every production asset."""
    rows = get_all_latest_predictions(serving_db)
    return [_to_prediction_out(r) for r in rows]


@router.get("/predictions/history", response_model=list[PredictionOut])
def predictions_history(
    asset_id: str = Query(..., description="Production asset ID"),
    limit: int = Query(90, ge=1, le=365, description="Number of rows (max 365)"),
    serving_db: ServingDB = Depends(get_serving_db),
):
    """Return historical predictions for one asset (newest first)."""
    rows = get_prediction_history(serving_db, asset_id, limit=limit)
    if not rows and get_latest_prediction(serving_db, asset_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No predictions found for asset '{asset_id}'.",
        )
    return [_to_prediction_out(r) for r in rows]


@router.get("/prices/history", response_model=list[PricePoint])
def prices_history(
    asset_id: str = Query(..., description="Production asset ID"),
    days: int = Query(180, ge=30, le=730, description="Lookback window in calendar days"),
    research_db: duckdb.DuckDBPyConnection = Depends(get_research_db),
):
    """Return daily close prices for one asset from the research DB."""
    try:
        catalog = load_asset_catalog(_ASSETS_CONFIG)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset '{asset_id}' not found.",
        )

    asset = next((a for a in catalog if a.asset_id == asset_id), None)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset '{asset_id}' not found.",
        )

    since = date.today() - timedelta(days=days)
    rows = research_db.execute(
        """
        SELECT CAST(date AS VARCHAR) AS date, close
        FROM raw_prices
        WHERE ticker = ? AND date >= ? AND close IS NOT NULL
        ORDER BY date ASC
        """,
        [asset.source_ticker, since],
    ).fetchall()

    return [PricePoint(date=r[0], close=r[1]) for r in rows]
