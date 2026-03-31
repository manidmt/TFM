'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Read-only ops queries for the admin /ops view.

All functions in this module are read-only — they never modify serving.duckdb.
They power the admin-only /ops/* route zone described in rpi5.md §12.3.

Available queries
-----------------
get_asset_status_all        — latest operational state per asset
get_ops_summary             — combined dashboard dict (counts + per-asset status)
get_recent_runs             — latest prediction_runs rows (newest first)
get_promotion_history       — recent promotion_events rows (newest first)
get_active_bundles          — current active bundle per asset

Reference: rpi5.md §11.2, §12.3, §13.1
'''

from __future__ import annotations

from typing import Any

from quant_risk.prod.serving.duckdb import ServingDB


# ---------------------------------------------------------------------------
# Asset status
# ---------------------------------------------------------------------------

def get_asset_status_all(db: ServingDB) -> list[dict[str, Any]]:
    """Return the latest operational status for every asset.

    Reads from ``asset_status``, which is refreshed on every completed batch
    attempt (success or failure) by runs.py.

    Returns rows ordered by asset_id ascending.
    """
    conn = db.connect()
    rows = conn.execute(
        """
        SELECT asset_id, last_forecast_date, last_run_status,
               last_run_at, active_bundle_version, updated_at
        FROM asset_status
        ORDER BY asset_id
        """
    ).fetchall()
    _cols = [
        "asset_id", "last_forecast_date", "last_run_status",
        "last_run_at", "active_bundle_version", "updated_at",
    ]
    return [dict(zip(_cols, r)) for r in rows]


# ---------------------------------------------------------------------------
# Ops summary (dashboard)
# ---------------------------------------------------------------------------

def get_ops_summary(db: ServingDB) -> dict[str, Any]:
    """Return a combined ops summary dict for the /ops dashboard.

    Structure
    ---------
    {
        "asset_count": int,
        "fresh_count": int,
        "stale_count": int,
        "failed_count": int,
        "assets": [
            {
                "asset_id": str,
                "last_forecast_date": date | None,
                "last_run_status": str | None,
                "last_run_at": str | None,
                "active_bundle_version": str | None,
            },
            ...
        ]
    }
    """
    assets = get_asset_status_all(db)

    fresh_count = sum(
        1 for a in assets if a.get("last_run_status") == "fresh"
    )
    stale_statuses = {"stale_news", "stale_data"}
    stale_count = sum(
        1 for a in assets if a.get("last_run_status") in stale_statuses
    )
    failed_count = sum(
        1 for a in assets if a.get("last_run_status") == "failed"
    )

    return {
        "asset_count": len(assets),
        "fresh_count": fresh_count,
        "stale_count": stale_count,
        "failed_count": failed_count,
        "assets": assets,
    }


# ---------------------------------------------------------------------------
# Recent prediction runs
# ---------------------------------------------------------------------------

def get_recent_runs(
    db: ServingDB,
    limit: int = 50,
    asset_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent prediction_runs rows, newest first.

    Parameters
    ----------
    db:
        Open ServingDB connection.
    limit:
        Maximum rows to return.
    asset_id:
        Optional filter to one asset.
    """
    conn = db.connect()
    _cols = [
        "run_id", "forecast_date", "asset_id", "attempt_no",
        "status", "error_code", "error_message",
        "started_at", "finished_at", "bundle_version",
    ]
    if asset_id is not None:
        rows = conn.execute(
            """
            SELECT run_id, forecast_date, asset_id, attempt_no,
                   status, error_code, error_message,
                   started_at, finished_at, bundle_version
            FROM prediction_runs
            WHERE asset_id = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            [asset_id, int(limit)],
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT run_id, forecast_date, asset_id, attempt_no,
                   status, error_code, error_message,
                   started_at, finished_at, bundle_version
            FROM prediction_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            [int(limit)],
        ).fetchall()
    return [dict(zip(_cols, r)) for r in rows]


# ---------------------------------------------------------------------------
# Promotion history
# ---------------------------------------------------------------------------

def get_promotion_history(
    db: ServingDB,
    limit: int = 20,
    asset_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent promotion_events rows, newest first.

    Parameters
    ----------
    db:
        Open ServingDB connection.
    limit:
        Maximum rows to return.
    asset_id:
        Optional filter to one asset.
    """
    conn = db.connect()
    _cols = [
        "event_id", "asset_id", "release_id", "bundle_version",
        "status", "previous_version", "validation_errors", "promoted_at",
    ]
    if asset_id is not None:
        rows = conn.execute(
            """
            SELECT event_id, asset_id, release_id, bundle_version,
                   status, previous_version, validation_errors, promoted_at
            FROM promotion_events
            WHERE asset_id = ?
            ORDER BY promoted_at DESC
            LIMIT ?
            """,
            [asset_id, int(limit)],
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT event_id, asset_id, release_id, bundle_version,
                   status, previous_version, validation_errors, promoted_at
            FROM promotion_events
            ORDER BY promoted_at DESC
            LIMIT ?
            """,
            [int(limit)],
        ).fetchall()
    return [dict(zip(_cols, r)) for r in rows]


# ---------------------------------------------------------------------------
# Active bundles
# ---------------------------------------------------------------------------

def get_active_bundles(db: ServingDB) -> list[dict[str, Any]]:
    """Return the currently active bundle for every asset.

    Reads from ``active_bundles``, which is updated on successful promotion.
    Returns rows ordered by asset_id ascending.
    """
    conn = db.connect()
    _cols = [
        "asset_id", "source_ticker", "release_id", "bundle_version",
        "model_type", "feature_profile", "promoted_at", "previous_bundle_version",
    ]
    rows = conn.execute(
        """
        SELECT asset_id, source_ticker, release_id, bundle_version,
               model_type, feature_profile, promoted_at, previous_bundle_version
        FROM active_bundles
        ORDER BY asset_id
        """
    ).fetchall()
    return [dict(zip(_cols, r)) for r in rows]
