'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-04-09

@description: Internal prediction push endpoint (Plan B / off-box inference).

Background
----------
TabPFN inference on the RPi5 CPU takes ~1–2 hours per asset and keeps hitting
memory / CUDA-stub edge cases. Rather than keep fighting the platform, the
production lane is split in two:

  * **Workstation** runs the full inference pipeline (ingest, features,
    bundle load, predict, calibrate) for every asset and POSTs the resulting
    predictions here.

  * **RPi5** keeps the public read API, nginx + Cloudflare Tunnel, and the
    auth/portfolio Postgres database. The worker container can be shut down
    entirely (or reduced to a shared-ingest-only run).

Security
--------
This router is protected by ``require_internal_token`` — a shared bearer token
read from ``QUANT_RISK_INTERNAL_TOKEN``. If the env var is unset, the endpoint
returns 503 (not 404 or 401) to make the "not configured" state unambiguous
in logs. See ``deps.py`` for details.

Route
-----
POST /api/internal/predictions
    Body: PredictionPushIn (see below).
    On success: upserts one row in ``predictions_daily``, writes a
    synthetic ``prediction_runs`` attempt, and refreshes ``asset_status``.
    Returns the stored row echoed back as ``PredictionOut``.

Idempotency
-----------
``(asset_id, forecast_date)`` is the logical primary key of
``predictions_daily`` — re-pushing the same tuple overwrites the previous row.
``prediction_runs`` is append-only: each push gets a new row with an
incrementing ``attempt_no`` so the audit trail is preserved.
'''

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from quant_risk.prod.api.deps import (
    get_serving_db,
    get_serving_db_rw,
    require_internal_token,
)
from quant_risk.prod.schemas import PredictedClass, RunStatus
from quant_risk.prod.serving.duckdb import ServingDB
from quant_risk.prod.serving.predictions import persist_prediction
from quant_risk.prod.worker.inference import PredictionOutput
from quant_risk.prod.worker.runs import (
    RunContext,
    RunResult,
    finish_run,
    start_run,
    update_asset_status,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/internal", tags=["internal"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_VALID_RUN_STATUSES = {s.value for s in RunStatus} - {RunStatus.failed.value}
_VALID_CLASSES = {c.value for c in PredictedClass}


class PredictionPushIn(BaseModel):
    """Prediction payload POSTed by the off-box inference runner.

    All probability fields are 0–1 floats that must sum to ~1.0 (tolerance ±0.02).
    """

    asset_id: str = Field(..., min_length=1, max_length=64)
    forecast_date: date
    predicted_class: Literal["low", "medium", "high"]
    p_low: float = Field(..., ge=0.0, le=1.0)
    p_medium: float = Field(..., ge=0.0, le=1.0)
    p_high: float = Field(..., ge=0.0, le=1.0)
    bundle_version: str = Field(..., min_length=1, max_length=128)
    data_cutoff_date: date
    run_status: Literal["fresh", "stale_news", "stale_data"] = "fresh"
    # Optional diagnostics — not stored, logged only.
    source_host: str | None = Field(default=None, max_length=128)

    @field_validator("p_low", "p_medium", "p_high")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):  # NaN or ±inf
            raise ValueError("probability must be a finite real number")
        return v


class PredictionPushOut(BaseModel):
    asset_id: str
    forecast_date: str
    predicted_class: str
    p_low: float
    p_medium: float
    p_high: float
    bundle_version: str
    data_cutoff_date: str
    run_status: str
    attempt_no: int
    stored_at: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/predictions",
    response_model=PredictionPushOut,
    status_code=status.HTTP_200_OK,
)
def push_prediction(
    payload: PredictionPushIn,
    serving_db: ServingDB = Depends(get_serving_db_rw),
    _token: None = Depends(require_internal_token),
) -> PredictionPushOut:
    """Persist a prediction produced off-box.

    The call writes atomically to ``predictions_daily``, ``prediction_runs``
    and ``asset_status`` inside a single DuckDB transaction so partial failures
    do not leave orphan rows in the audit trail.
    """
    # --- 1. Sanity checks that Pydantic can't express alone. -------------
    total = payload.p_low + payload.p_medium + payload.p_high
    if not (0.98 <= total <= 1.02):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Class probabilities must sum to ~1.0, got {total:.4f}.",
        )

    # argmax vs reported class — allow client to override (client ran the
    # threshold-gated selector), but flag mismatches in the logs.
    probs = {"low": payload.p_low, "medium": payload.p_medium, "high": payload.p_high}
    argmax_class = max(probs, key=probs.get)  # type: ignore[arg-type]
    if argmax_class != payload.predicted_class:
        logger.info(
            "[%s] predicted_class=%s differs from argmax=%s (thresholds may have applied)",
            payload.asset_id, payload.predicted_class, argmax_class,
        )

    if payload.forecast_date > date.today():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="forecast_date cannot be in the future.",
        )
    if payload.data_cutoff_date > payload.forecast_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="data_cutoff_date must be <= forecast_date.",
        )

    run_status_enum = RunStatus(payload.run_status)
    prediction = PredictionOutput(
        predicted_class=PredictedClass(payload.predicted_class),
        p_low=payload.p_low,
        p_medium=payload.p_medium,
        p_high=payload.p_high,
        bundle_version=payload.bundle_version,
    )

    # --- 2. Persist inside a single transaction. -------------------------
    conn = serving_db.connect()
    conn.execute("BEGIN")
    try:
        ctx: RunContext = start_run(
            serving_db, payload.asset_id, payload.forecast_date,
        )

        stored_at = datetime.now(timezone.utc)
        persist_prediction(
            db=serving_db,
            asset_id=payload.asset_id,
            forecast_date=payload.forecast_date,
            prediction=prediction,
            status=run_status_enum,
            data_cutoff_date=payload.data_cutoff_date,
            updated_at=stored_at,
        )

        result = RunResult(
            status=run_status_enum,
            bundle_version=prediction.bundle_version,
            finished_at=stored_at,
        )
        finish_run(serving_db, ctx, result)
        update_asset_status(serving_db, ctx, result)

        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:  # noqa: BLE001
            pass
        raise

    logger.info(
        "[%s] pushed prediction forecast_date=%s class=%s bundle=%s source=%s attempt=%d",
        payload.asset_id,
        payload.forecast_date,
        payload.predicted_class,
        payload.bundle_version,
        payload.source_host or "?",
        ctx.attempt_no,
    )

    return PredictionPushOut(
        asset_id=payload.asset_id,
        forecast_date=payload.forecast_date.isoformat(),
        predicted_class=payload.predicted_class,
        p_low=payload.p_low,
        p_medium=payload.p_medium,
        p_high=payload.p_high,
        bundle_version=payload.bundle_version,
        data_cutoff_date=payload.data_cutoff_date.isoformat(),
        run_status=payload.run_status,
        attempt_no=ctx.attempt_no,
        stored_at=stored_at.isoformat(),
    )
