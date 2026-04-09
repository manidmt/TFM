'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: serving.duckdb connection management and schema initialisation.

serving.duckdb is the operational data store for the production lane on the RPi5.
It holds prediction history, active bundle state, run metadata, and promotion
events.  It is intentionally separate from financial_data.duckdb to decouple
write-heavy batch work (ingestion, feature build) from read-heavy app traffic
(API serving predictions) — see rpi5.md §11.4.

Schema design reference: rpi5.md §11.2, §26.5

Tables
------
active_bundles     : one row per asset_id — the currently promoted bundle
predictions_daily  : one logical row per (asset_id, forecast_date) — idempotent
prediction_runs    : attempt-level log for every batch run per asset
promotion_events   : full audit trail of every promotion attempt
asset_status       : latest operational state per asset (refreshed each run)

Usage
-----
    db = ServingDB("/srv/quant-risk/db/serving.duckdb")
    db.create_schema()          # idempotent — safe to call on every startup
    conn = db.connect()         # raw duckdb connection for ad-hoc queries
    db.close()
'''

from __future__ import annotations

import os
from pathlib import Path

import duckdb


# DDL for each table.  All CREATE statements use IF NOT EXISTS so the call is
# idempotent and safe to run on every worker startup.

_DDL_ACTIVE_BUNDLES = """
CREATE TABLE IF NOT EXISTS active_bundles (
    asset_id              VARCHAR PRIMARY KEY,
    source_ticker         VARCHAR NOT NULL,
    release_id            VARCHAR NOT NULL,
    bundle_version        VARCHAR NOT NULL,
    model_type            VARCHAR NOT NULL,
    feature_profile       VARCHAR NOT NULL,
    promoted_at           TIMESTAMPTZ NOT NULL,
    previous_bundle_version VARCHAR
);
"""

_DDL_PREDICTIONS_DAILY = """
CREATE TABLE IF NOT EXISTS predictions_daily (
    asset_id         VARCHAR  NOT NULL,
    forecast_date    DATE     NOT NULL,
    predicted_class  VARCHAR  NOT NULL,    -- 'low' | 'medium' | 'high'
    p_low            DOUBLE   NOT NULL,
    p_medium         DOUBLE   NOT NULL,
    p_high           DOUBLE   NOT NULL,
    bundle_version   VARCHAR  NOT NULL,
    data_cutoff_date DATE     NOT NULL,    -- last market date used
    status           VARCHAR  NOT NULL,    -- RunStatus enum value
    updated_at       TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (asset_id, forecast_date)
);
"""

_DDL_PREDICTION_RUNS = """
CREATE TABLE IF NOT EXISTS prediction_runs (
    run_id        VARCHAR PRIMARY KEY,
    forecast_date DATE    NOT NULL,
    asset_id      VARCHAR NOT NULL,
    attempt_no    INTEGER NOT NULL DEFAULT 1,
    status        VARCHAR NOT NULL,    -- RunStatus enum value
    error_code    VARCHAR,
    error_message VARCHAR,
    started_at    TIMESTAMPTZ NOT NULL,
    finished_at   TIMESTAMPTZ,
    bundle_version VARCHAR
);
CREATE INDEX IF NOT EXISTS idx_prediction_runs_asset_date
    ON prediction_runs (asset_id, forecast_date);
"""

_DDL_PROMOTION_EVENTS = """
CREATE TABLE IF NOT EXISTS promotion_events (
    event_id          VARCHAR PRIMARY KEY,
    asset_id          VARCHAR NOT NULL,
    release_id        VARCHAR NOT NULL,
    bundle_version    VARCHAR NOT NULL,
    status            VARCHAR NOT NULL,    -- PromotionStatus enum value
    previous_version  VARCHAR,
    validation_errors VARCHAR,             -- JSON array of error strings, or NULL
    promoted_at       TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_promotion_events_asset
    ON promotion_events (asset_id, promoted_at);
"""

_DDL_ASSET_STATUS = """
CREATE TABLE IF NOT EXISTS asset_status (
    asset_id          VARCHAR PRIMARY KEY,
    last_forecast_date DATE,
    last_run_status   VARCHAR,             -- RunStatus enum value
    last_run_at       TIMESTAMPTZ,
    active_bundle_version VARCHAR,
    updated_at        TIMESTAMPTZ NOT NULL
);
"""

_ALL_DDL: list[str] = [
    _DDL_ACTIVE_BUNDLES,
    _DDL_PREDICTIONS_DAILY,
    _DDL_PREDICTION_RUNS,
    _DDL_PROMOTION_EVENTS,
    _DDL_ASSET_STATUS,
]


class ServingDB:
    """Thin wrapper around a serving.duckdb connection.

    Parameters
    ----------
    path:
        Path to the DuckDB file.  Resolved at runtime from the
        QUANT_RISK_SERVING_DB_PATH environment variable if set, otherwise the
        value passed here is used.  Defaults to the RPi5 canonical path.
    read_only:
        Open the database in read-only mode (useful for the API container
        which should never write directly to serving.duckdb).
    """

    DEFAULT_PATH = "/srv/quant-risk/db/serving.duckdb"

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        resolved = os.environ.get("QUANT_RISK_SERVING_DB_PATH") or path or self.DEFAULT_PATH
        self._path = Path(resolved)
        self._read_only = read_only
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Return the active connection, opening it if necessary."""
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(
                str(self._path),
                read_only=self._read_only,
            )
            # Workaround for a DuckDB optimizer assertion failure observed on
            # 1.5.1 / aarch64: when prediction_runs receives an INSERT and the
            # next MAX(attempt_no) aggregate reuses in-process partition
            # statistics, StatisticsPropagator::TryExecuteAggregates crashes
            # with "Attempted to access index 0 within vector of size 0".
            # Disabling the propagator has no functional impact for our tiny
            # tables (a few thousand rows) and sidesteps the bug.
            if not self._read_only:
                try:
                    self._conn.execute(
                        "SET disabled_optimizers='statistics_propagation'"
                    )
                except Exception:  # noqa: BLE001
                    # Older DuckDB versions may not recognise the pragma —
                    # silently ignore rather than break.
                    pass
        return self._conn

    def close(self) -> None:
        """Close the connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def ping(self) -> bool:
        """Return True if the serving DB is reachable."""
        try:
            self.connect().execute("SELECT 1")
            return True
        except Exception:
            return False

    def __enter__(self) -> "ServingDB":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """Create all serving.duckdb tables (idempotent — safe on every startup).

        Each DDL block uses CREATE TABLE IF NOT EXISTS so calling this multiple
        times is safe.  Indices are also created with IF NOT EXISTS.
        """
        conn = self.connect()
        for ddl in _ALL_DDL:
            # DuckDB 0.9.x executes multi-statement strings correctly via executescript.
            # Split on semicolons so individual statements work on all versions.
            for stmt in _split_statements(ddl):
                if stmt:
                    conn.execute(stmt)

    def table_names(self) -> list[str]:
        """Return the list of table names present in serving.duckdb."""
        conn = self.connect()
        result = conn.execute("SHOW TABLES").fetchall()
        return [row[0] for row in result]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_statements(ddl: str) -> list[str]:
    """Split a multi-statement DDL block into individual statements."""
    return [s.strip() for s in ddl.split(";") if s.strip()]
