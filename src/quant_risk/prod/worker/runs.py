'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Per-asset run state tracking and attempt logging for the production worker.

Every daily batch execution is logged at the attempt level in prediction_runs.
Multiple attempts are allowed for the same (asset_id, forecast_date) — each
gets its own run_id and an incrementing attempt_no.

The asset_status table keeps only the latest operational state per asset and
is refreshed on every completed attempt (success or failure).

Reference: rpi5.md §10.4, §10.5, §11.2, §26.5
'''

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from quant_risk.prod.schemas import RunStatus
from quant_risk.prod.serving.duckdb import ServingDB


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    """Identifies one attempt of the daily batch for a single asset."""

    run_id: str
    asset_id: str
    forecast_date: date
    attempt_no: int
    started_at: datetime


@dataclass
class RunResult:
    """Outcome of one attempt of the daily batch for a single asset."""

    status: RunStatus
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    finished_at: Optional[datetime] = None
    bundle_version: Optional[str] = None

    def is_successful(self) -> bool:
        return self.status != RunStatus.failed


# ---------------------------------------------------------------------------
# Run lifecycle helpers
# ---------------------------------------------------------------------------

def start_run(
    db: ServingDB,
    asset_id: str,
    forecast_date: date,
) -> RunContext:
    """Open a new attempt in prediction_runs and return its RunContext.

    The attempt_no is auto-incremented based on any existing rows for the
    same (asset_id, forecast_date).  This supports the idempotent-rerun
    policy described in rpi5.md §10.5.
    """
    conn = db.connect()
    row = conn.execute(
        "SELECT COALESCE(MAX(attempt_no), 0) FROM prediction_runs "
        "WHERE asset_id = ? AND forecast_date = ?",
        [asset_id, forecast_date.isoformat()],
    ).fetchone()
    attempt_no: int = (row[0] if row else 0) + 1

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    conn.execute(
        """
        INSERT INTO prediction_runs
            (run_id, forecast_date, asset_id, attempt_no, status, started_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            forecast_date.isoformat(),
            asset_id,
            attempt_no,
            RunStatus.failed.value,   # default — overwritten by finish_run
            started_at.isoformat(),
        ],
    )

    return RunContext(
        run_id=run_id,
        asset_id=asset_id,
        forecast_date=forecast_date,
        attempt_no=attempt_no,
        started_at=started_at,
    )


def finish_run(
    db: ServingDB,
    ctx: RunContext,
    result: RunResult,
) -> None:
    """Update the prediction_runs row with the final outcome of an attempt."""
    finished_at = result.finished_at or datetime.now(timezone.utc)
    conn = db.connect()
    conn.execute(
        """
        UPDATE prediction_runs
        SET status        = ?,
            error_code    = ?,
            error_message = ?,
            finished_at   = ?,
            bundle_version = ?
        WHERE run_id = ?
        """,
        [
            result.status.value,
            result.error_code,
            result.error_message,
            finished_at.isoformat(),
            result.bundle_version,
            ctx.run_id,
        ],
    )


def update_asset_status(
    db: ServingDB,
    ctx: RunContext,
    result: RunResult,
) -> None:
    """Upsert asset_status for *ctx.asset_id* with the latest run outcome.

    Uses DELETE + INSERT for compatibility with DuckDB 0.9.x which does not
    support the INSERT … ON CONFLICT DO UPDATE syntax uniformly.
    """
    updated_at = datetime.now(timezone.utc)
    finished_at = result.finished_at or updated_at
    conn = db.connect()

    conn.execute("DELETE FROM asset_status WHERE asset_id = ?", [ctx.asset_id])
    conn.execute(
        """
        INSERT INTO asset_status
            (asset_id, last_forecast_date, last_run_status, last_run_at,
             active_bundle_version, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            ctx.asset_id,
            ctx.forecast_date.isoformat(),
            result.status.value,
            finished_at.isoformat(),
            result.bundle_version,
            updated_at.isoformat(),
        ],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_failed_result(
    error_message: str,
    error_code: str = "UNKNOWN_ERROR",
) -> RunResult:
    """Convenience factory for a failed RunResult."""
    return RunResult(
        status=RunStatus.failed,
        error_code=error_code,
        error_message=error_message,
        finished_at=datetime.now(timezone.utc),
    )
