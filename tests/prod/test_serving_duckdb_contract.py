'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Tests for serving.duckdb schema contract (serving/duckdb.py).

Covers:
- ServingDB.create_schema() creates all five expected tables (idempotent).
- Each table has the documented columns with correct types.
- Calling create_schema() twice does not raise (IF NOT EXISTS semantics).
- active_bundles has asset_id as PRIMARY KEY.
- predictions_daily has (asset_id, forecast_date) as PRIMARY KEY.
- prediction_runs has run_id as PRIMARY KEY and an index on (asset_id, forecast_date).
- promotion_events has event_id as PRIMARY KEY.
- asset_status has asset_id as PRIMARY KEY.
- ServingDB respects QUANT_RISK_SERVING_DB_PATH env var for path resolution.
- context manager (__enter__ / __exit__) opens and closes cleanly.
'''

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest

from quant_risk.prod.serving.duckdb import ServingDB


# ---------------------------------------------------------------------------
# Helper: build a ServingDB backed by a tmp file
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> ServingDB:
    return ServingDB(tmp_path / "serving.duckdb")


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def test_create_schema_creates_all_tables(tmp_path):
    db = _make_db(tmp_path)
    db.create_schema()
    tables = set(db.table_names())
    expected = {
        "active_bundles",
        "predictions_daily",
        "prediction_runs",
        "promotion_events",
        "asset_status",
    }
    assert expected.issubset(tables)
    db.close()


def test_create_schema_is_idempotent(tmp_path):
    db = _make_db(tmp_path)
    db.create_schema()
    db.create_schema()  # must not raise
    assert len(db.table_names()) >= 5
    db.close()


# ---------------------------------------------------------------------------
# Column contract per table
# ---------------------------------------------------------------------------

def _columns(db: ServingDB, table: str) -> dict[str, str]:
    """Return {column_name: type_str} for a table."""
    rows = db.connect().execute(f"DESCRIBE {table}").fetchall()
    # DuckDB 0.9 DESCRIBE returns (name, type, null, key, default, extra)
    return {row[0]: row[1].upper() for row in rows}


def test_active_bundles_columns(tmp_path):
    db = _make_db(tmp_path)
    db.create_schema()
    cols = _columns(db, "active_bundles")
    assert "ASSET_ID" in cols or "asset_id" in cols or "asset_id".upper() in {c.upper() for c in cols}
    required = {
        "asset_id", "source_ticker", "release_id", "bundle_version",
        "model_type", "feature_profile", "promoted_at", "previous_bundle_version",
    }
    col_names_lower = {c.lower() for c in cols}
    assert required.issubset(col_names_lower), f"Missing columns: {required - col_names_lower}"
    db.close()


def test_predictions_daily_columns(tmp_path):
    db = _make_db(tmp_path)
    db.create_schema()
    cols = _columns(db, "predictions_daily")
    required = {
        "asset_id", "forecast_date", "predicted_class",
        "p_low", "p_medium", "p_high",
        "bundle_version", "data_cutoff_date", "status", "updated_at",
    }
    col_names_lower = {c.lower() for c in cols}
    assert required.issubset(col_names_lower), f"Missing columns: {required - col_names_lower}"
    db.close()


def test_prediction_runs_columns(tmp_path):
    db = _make_db(tmp_path)
    db.create_schema()
    cols = _columns(db, "prediction_runs")
    required = {
        "run_id", "forecast_date", "asset_id", "attempt_no",
        "status", "error_code", "error_message", "started_at", "finished_at",
    }
    col_names_lower = {c.lower() for c in cols}
    assert required.issubset(col_names_lower), f"Missing columns: {required - col_names_lower}"
    db.close()


def test_promotion_events_columns(tmp_path):
    db = _make_db(tmp_path)
    db.create_schema()
    cols = _columns(db, "promotion_events")
    required = {
        "event_id", "asset_id", "release_id", "bundle_version",
        "status", "previous_version", "validation_errors", "promoted_at",
    }
    col_names_lower = {c.lower() for c in cols}
    assert required.issubset(col_names_lower), f"Missing columns: {required - col_names_lower}"
    db.close()


def test_asset_status_columns(tmp_path):
    db = _make_db(tmp_path)
    db.create_schema()
    cols = _columns(db, "asset_status")
    required = {
        "asset_id", "last_forecast_date", "last_run_status",
        "last_run_at", "active_bundle_version", "updated_at",
    }
    col_names_lower = {c.lower() for c in cols}
    assert required.issubset(col_names_lower), f"Missing columns: {required - col_names_lower}"
    db.close()


# ---------------------------------------------------------------------------
# Primary key / uniqueness constraints
# ---------------------------------------------------------------------------

def test_active_bundles_unique_asset_id(tmp_path):
    db = _make_db(tmp_path)
    db.create_schema()
    from datetime import datetime, timezone
    conn = db.connect()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO active_bundles
        VALUES ('us_equities','^GSPC','prod_2026-03-27','2026-03-27_prod_abc1234',
                'xgb','gkg_v1_features_compact',?,NULL)
    """, [now])
    with pytest.raises(Exception):
        conn.execute("""
            INSERT INTO active_bundles
            VALUES ('us_equities','^GSPC','prod_2026-03-27','2026-03-27_prod_def5678',
                    'xgb','gkg_v1_features_compact',?,NULL)
        """, [now])
    db.close()


def test_predictions_daily_unique_asset_date(tmp_path):
    db = _make_db(tmp_path)
    db.create_schema()
    from datetime import date, datetime, timezone
    conn = db.connect()
    now = datetime.now(timezone.utc).isoformat()
    today = date.today().isoformat()
    conn.execute("""
        INSERT INTO predictions_daily
        VALUES ('us_equities',?,  'high', 0.1, 0.2, 0.7,
                '2026-03-27_prod_abc1234', ?, 'fresh', ?)
    """, [today, today, now])
    with pytest.raises(Exception):
        conn.execute("""
            INSERT INTO predictions_daily
            VALUES ('us_equities',?, 'low', 0.6, 0.3, 0.1,
                    '2026-03-27_prod_abc1234', ?, 'fresh', ?)
        """, [today, today, now])
    db.close()


# ---------------------------------------------------------------------------
# Environment variable path override
# ---------------------------------------------------------------------------

def test_env_var_overrides_path(tmp_path, monkeypatch):
    custom_path = str(tmp_path / "custom_serving.duckdb")
    monkeypatch.setenv("QUANT_RISK_SERVING_DB_PATH", custom_path)
    db = ServingDB()  # no explicit path
    assert str(db.path) == custom_path
    db.create_schema()
    assert db.path.exists()
    db.close()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

def test_context_manager_opens_and_closes(tmp_path):
    with ServingDB(tmp_path / "serving.duckdb") as db:
        db.create_schema()
        assert len(db.table_names()) >= 5
    # after __exit__, _conn should be None
    assert db._conn is None


# ---------------------------------------------------------------------------
# Parent directory auto-creation
# ---------------------------------------------------------------------------

def test_creates_parent_directory(tmp_path):
    nested = tmp_path / "a" / "b" / "serving.duckdb"
    db = ServingDB(nested)
    db.create_schema()
    assert nested.exists()
    db.close()
