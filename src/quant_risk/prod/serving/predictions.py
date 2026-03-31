'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Prediction persistence and read helpers for serving.duckdb.

Write side
----------
persist_prediction — idempotent upsert of one predictions_daily row.

Idempotency contract (rpi5.md §10.5):
  - One logical final prediction row per (asset_id, forecast_date).
  - Re-running for the same pair replaces the previous row.
  - Attempt-level logging (prediction_runs) is handled by runs.py, not here.
  - Upsert uses DELETE + INSERT for DuckDB 0.9 compatibility
    (no ON CONFLICT … DO UPDATE support uniformly).

Read side
---------
get_latest_prediction   — most recent prediction for one asset
get_all_latest_predictions — latest prediction per asset across all assets
get_prediction_history  — historical predictions for one asset (newest first)

Reference: rpi5.md §10.5, §11.2, §26.5
'''

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from quant_risk.prod.schemas import RunStatus
from quant_risk.prod.serving.duckdb import ServingDB
from quant_risk.prod.worker.inference import PredictionOutput


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def persist_prediction(
    db: ServingDB,
    asset_id: str,
    forecast_date: date,
    prediction: PredictionOutput,
    status: RunStatus,
    data_cutoff_date: date,
    updated_at: datetime | None = None,
) -> None:
    """Upsert one prediction row into predictions_daily.

    Idempotent: re-running for the same (asset_id, forecast_date) replaces the
    previous row.  The attempt-level log in prediction_runs is NOT touched here
    — that is handled by runs.py.

    Parameters
    ----------
    db:
        Open ServingDB connection.
    asset_id:
        Semantic asset identifier.
    forecast_date:
        The D-1 date for which the prediction was produced.
    prediction:
        Output from run_inference_for_asset.
    status:
        RunStatus describing data quality (fresh / stale_news / stale_data).
    data_cutoff_date:
        Last market date used to build features.
    updated_at:
        Override for the updated_at timestamp (defaults to now UTC).
    """
    ts = updated_at or datetime.now(timezone.utc)
    conn = db.connect()

    # DELETE + INSERT: DuckDB 0.9 upsert pattern (used throughout the codebase)
    conn.execute(
        "DELETE FROM predictions_daily WHERE asset_id = ? AND forecast_date = ?",
        [asset_id, forecast_date.isoformat()],
    )
    conn.execute(
        """
        INSERT INTO predictions_daily
            (asset_id, forecast_date, predicted_class,
             p_low, p_medium, p_high,
             bundle_version, data_cutoff_date, status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            asset_id,
            forecast_date.isoformat(),
            prediction.predicted_class.value,
            float(prediction.p_low),
            float(prediction.p_medium),
            float(prediction.p_high),
            prediction.bundle_version,
            data_cutoff_date.isoformat(),
            status.value,
            ts.isoformat(),
        ],
    )


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_latest_prediction(
    db: ServingDB,
    asset_id: str,
) -> dict[str, Any] | None:
    """Return the most recent prediction row for *asset_id*, or None.

    "Most recent" is determined by the largest forecast_date.
    """
    conn = db.connect()
    row = conn.execute(
        """
        SELECT asset_id, forecast_date, predicted_class,
               p_low, p_medium, p_high,
               bundle_version, data_cutoff_date, status, updated_at
        FROM predictions_daily
        WHERE asset_id = ?
        ORDER BY forecast_date DESC
        LIMIT 1
        """,
        [asset_id],
    ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def get_all_latest_predictions(db: ServingDB) -> list[dict[str, Any]]:
    """Return the most recent prediction row for every asset.

    One row per asset_id, ordered by asset_id ascending.
    """
    conn = db.connect()
    rows = conn.execute(
        """
        SELECT p.asset_id, p.forecast_date, p.predicted_class,
               p.p_low, p.p_medium, p.p_high,
               p.bundle_version, p.data_cutoff_date, p.status, p.updated_at
        FROM predictions_daily p
        INNER JOIN (
            SELECT asset_id, MAX(forecast_date) AS max_date
            FROM predictions_daily
            GROUP BY asset_id
        ) latest ON p.asset_id = latest.asset_id
                AND p.forecast_date = latest.max_date
        ORDER BY p.asset_id
        """
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_prediction_history(
    db: ServingDB,
    asset_id: str,
    limit: int = 90,
) -> list[dict[str, Any]]:
    """Return historical predictions for *asset_id*, newest first.

    Parameters
    ----------
    db:
        Open ServingDB connection.
    asset_id:
        Semantic asset identifier.
    limit:
        Maximum number of rows to return (default 90 — ~3 months of daily data).
    """
    conn = db.connect()
    rows = conn.execute(
        """
        SELECT asset_id, forecast_date, predicted_class,
               p_low, p_medium, p_high,
               bundle_version, data_cutoff_date, status, updated_at
        FROM predictions_daily
        WHERE asset_id = ?
        ORDER BY forecast_date DESC
        LIMIT ?
        """,
        [asset_id, int(limit)],
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

_COLUMNS = [
    "asset_id", "forecast_date", "predicted_class",
    "p_low", "p_medium", "p_high",
    "bundle_version", "data_cutoff_date", "status", "updated_at",
]


def _row_to_dict(row: tuple) -> dict[str, Any]:
    """Map a predictions_daily tuple to a dict."""
    return dict(zip(_COLUMNS, row))
