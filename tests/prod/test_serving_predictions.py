"""
Tests for src/quant_risk/prod/serving/predictions.py and serving/ops.py.

Coverage:
- persist_prediction: happy path, idempotent overwrite, status values
- get_latest_prediction: present / absent
- get_all_latest_predictions: empty DB, one asset, multiple assets
- get_prediction_history: limit, ordering, unknown asset
- ops.get_asset_status_all: empty / populated
- ops.get_ops_summary: counts
- ops.get_recent_runs: ordering, asset filter
- ops.get_promotion_history: asset filter
- ops.get_active_bundles: populated
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from quant_risk.prod.schemas import PredictedClass, PromotionStatus, RunStatus
from quant_risk.prod.serving.duckdb import ServingDB
from quant_risk.prod.serving.ops import (
    get_active_bundles,
    get_asset_status_all,
    get_ops_summary,
    get_promotion_history,
    get_recent_runs,
)
from quant_risk.prod.serving.predictions import (
    get_all_latest_predictions,
    get_latest_prediction,
    get_prediction_history,
    persist_prediction,
)
from quant_risk.prod.worker.inference import PredictionOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path):
    """Fresh in-memory-equivalent ServingDB backed by a tmp file."""
    path = str(tmp_path / "serving.duckdb")
    sdb = ServingDB(path)
    sdb.create_schema()
    yield sdb
    sdb.close()


def _prediction(proba=(0.2, 0.3, 0.5), bundle="2026-03-27_prod_abc1234"):
    p_low, p_medium, p_high = proba
    predicted = PredictedClass.high if p_high > p_medium else PredictedClass.medium
    return PredictionOutput(
        predicted_class=predicted,
        p_low=p_low,
        p_medium=p_medium,
        p_high=p_high,
        bundle_version=bundle,
    )


# ---------------------------------------------------------------------------
# persist_prediction — write
# ---------------------------------------------------------------------------

class TestPersistPrediction:
    def test_inserts_row(self, db):
        persist_prediction(
            db=db,
            asset_id="us_equities",
            forecast_date=date(2024, 6, 28),
            prediction=_prediction(),
            status=RunStatus.fresh,
            data_cutoff_date=date(2024, 6, 28),
        )
        rows = db.connect().execute("SELECT * FROM predictions_daily").fetchall()
        assert len(rows) == 1

    def test_idempotent_overwrite(self, db):
        d = date(2024, 6, 28)
        persist_prediction(db=db, asset_id="us_equities", forecast_date=d,
                           prediction=_prediction((0.1, 0.2, 0.7)), status=RunStatus.fresh,
                           data_cutoff_date=d)
        # Second call with different proba
        persist_prediction(db=db, asset_id="us_equities", forecast_date=d,
                           prediction=_prediction((0.7, 0.2, 0.1)), status=RunStatus.fresh,
                           data_cutoff_date=d)
        rows = db.connect().execute(
            "SELECT p_low FROM predictions_daily WHERE asset_id='us_equities'"
        ).fetchall()
        assert len(rows) == 1
        # Second call should have overwritten — p_low is now 0.7
        assert rows[0][0] == pytest.approx(0.7)

    def test_different_dates_two_rows(self, db):
        for d in [date(2024, 6, 27), date(2024, 6, 28)]:
            persist_prediction(db=db, asset_id="us_equities", forecast_date=d,
                               prediction=_prediction(), status=RunStatus.fresh,
                               data_cutoff_date=d)
        count = db.connect().execute(
            "SELECT COUNT(*) FROM predictions_daily WHERE asset_id='us_equities'"
        ).fetchone()[0]
        assert count == 2

    def test_different_assets_two_rows(self, db):
        d = date(2024, 6, 28)
        for asset in ["us_equities", "bitcoin"]:
            persist_prediction(db=db, asset_id=asset, forecast_date=d,
                               prediction=_prediction(), status=RunStatus.fresh,
                               data_cutoff_date=d)
        count = db.connect().execute(
            "SELECT COUNT(*) FROM predictions_daily"
        ).fetchone()[0]
        assert count == 2

    def test_status_persisted(self, db):
        persist_prediction(db=db, asset_id="us_equities", forecast_date=date(2024, 6, 28),
                           prediction=_prediction(), status=RunStatus.stale_news,
                           data_cutoff_date=date(2024, 6, 28))
        row = db.connect().execute(
            "SELECT status FROM predictions_daily"
        ).fetchone()
        assert row[0] == "stale_news"

    def test_stale_data_status(self, db):
        persist_prediction(db=db, asset_id="us_equities", forecast_date=date(2024, 6, 28),
                           prediction=_prediction(), status=RunStatus.stale_data,
                           data_cutoff_date=date(2024, 6, 25))
        row = db.connect().execute(
            "SELECT status FROM predictions_daily"
        ).fetchone()
        assert row[0] == "stale_data"

    def test_bundle_version_stored(self, db):
        persist_prediction(db=db, asset_id="us_equities", forecast_date=date(2024, 6, 28),
                           prediction=_prediction(bundle="2026-03-27_prod_deadbeef"),
                           status=RunStatus.fresh, data_cutoff_date=date(2024, 6, 28))
        row = db.connect().execute(
            "SELECT bundle_version FROM predictions_daily"
        ).fetchone()
        assert row[0] == "2026-03-27_prod_deadbeef"

    def test_custom_updated_at(self, db):
        ts = datetime(2024, 6, 28, 12, 0, 0, tzinfo=timezone.utc)
        persist_prediction(db=db, asset_id="us_equities", forecast_date=date(2024, 6, 28),
                           prediction=_prediction(), status=RunStatus.fresh,
                           data_cutoff_date=date(2024, 6, 28), updated_at=ts)
        row = db.connect().execute(
            "SELECT updated_at FROM predictions_daily"
        ).fetchone()
        assert "2024-06-28" in str(row[0])


# ---------------------------------------------------------------------------
# get_latest_prediction
# ---------------------------------------------------------------------------

class TestGetLatestPrediction:
    def test_returns_none_when_empty(self, db):
        assert get_latest_prediction(db, "us_equities") is None

    def test_returns_row(self, db):
        d = date(2024, 6, 28)
        persist_prediction(db=db, asset_id="us_equities", forecast_date=d,
                           prediction=_prediction(), status=RunStatus.fresh,
                           data_cutoff_date=d)
        result = get_latest_prediction(db, "us_equities")
        assert result is not None
        assert result["asset_id"] == "us_equities"

    def test_returns_most_recent(self, db):
        for d in [date(2024, 6, 26), date(2024, 6, 27), date(2024, 6, 28)]:
            persist_prediction(db=db, asset_id="us_equities", forecast_date=d,
                               prediction=_prediction(), status=RunStatus.fresh,
                               data_cutoff_date=d)
        result = get_latest_prediction(db, "us_equities")
        assert result["forecast_date"] == date(2024, 6, 28)

    def test_isolates_by_asset(self, db):
        persist_prediction(db=db, asset_id="bitcoin", forecast_date=date(2024, 6, 28),
                           prediction=_prediction(), status=RunStatus.fresh,
                           data_cutoff_date=date(2024, 6, 28))
        assert get_latest_prediction(db, "us_equities") is None

    def test_dict_has_all_keys(self, db):
        d = date(2024, 6, 28)
        persist_prediction(db=db, asset_id="us_equities", forecast_date=d,
                           prediction=_prediction(), status=RunStatus.fresh,
                           data_cutoff_date=d)
        result = get_latest_prediction(db, "us_equities")
        expected_keys = {
            "asset_id", "forecast_date", "predicted_class",
            "p_low", "p_medium", "p_high",
            "bundle_version", "data_cutoff_date", "status", "updated_at",
        }
        assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# get_all_latest_predictions
# ---------------------------------------------------------------------------

class TestGetAllLatestPredictions:
    def test_empty_db(self, db):
        assert get_all_latest_predictions(db) == []

    def test_one_asset(self, db):
        persist_prediction(db=db, asset_id="us_equities", forecast_date=date(2024, 6, 28),
                           prediction=_prediction(), status=RunStatus.fresh,
                           data_cutoff_date=date(2024, 6, 28))
        result = get_all_latest_predictions(db)
        assert len(result) == 1
        assert result[0]["asset_id"] == "us_equities"

    def test_multiple_assets_one_row_each(self, db):
        for asset in ["bitcoin", "gold", "us_equities"]:
            for d in [date(2024, 6, 27), date(2024, 6, 28)]:
                persist_prediction(db=db, asset_id=asset, forecast_date=d,
                                   prediction=_prediction(), status=RunStatus.fresh,
                                   data_cutoff_date=d)
        result = get_all_latest_predictions(db)
        assert len(result) == 3
        for r in result:
            assert r["forecast_date"] == date(2024, 6, 28)

    def test_ordered_by_asset_id(self, db):
        for asset in ["us_equities", "bitcoin", "gold"]:
            persist_prediction(db=db, asset_id=asset, forecast_date=date(2024, 6, 28),
                               prediction=_prediction(), status=RunStatus.fresh,
                               data_cutoff_date=date(2024, 6, 28))
        result = get_all_latest_predictions(db)
        ids = [r["asset_id"] for r in result]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# get_prediction_history
# ---------------------------------------------------------------------------

class TestGetPredictionHistory:
    def test_empty_returns_empty_list(self, db):
        assert get_prediction_history(db, "us_equities") == []

    def test_ordered_newest_first(self, db):
        for d in [date(2024, 6, 26), date(2024, 6, 27), date(2024, 6, 28)]:
            persist_prediction(db=db, asset_id="us_equities", forecast_date=d,
                               prediction=_prediction(), status=RunStatus.fresh,
                               data_cutoff_date=d)
        history = get_prediction_history(db, "us_equities")
        dates = [r["forecast_date"] for r in history]
        assert dates == sorted(dates, reverse=True)

    def test_limit_respected(self, db):
        for i in range(10):
            d = date(2024, 1, i + 2)  # Jan 2 to Jan 11
            persist_prediction(db=db, asset_id="us_equities", forecast_date=d,
                               prediction=_prediction(), status=RunStatus.fresh,
                               data_cutoff_date=d)
        result = get_prediction_history(db, "us_equities", limit=5)
        assert len(result) == 5

    def test_unknown_asset_returns_empty(self, db):
        persist_prediction(db=db, asset_id="bitcoin", forecast_date=date(2024, 6, 28),
                           prediction=_prediction(), status=RunStatus.fresh,
                           data_cutoff_date=date(2024, 6, 28))
        assert get_prediction_history(db, "us_equities") == []


# ---------------------------------------------------------------------------
# ops — get_asset_status_all
# ---------------------------------------------------------------------------

class TestGetAssetStatusAll:
    def _seed_status(self, db, asset_id, status="fresh"):
        conn = db.connect()
        conn.execute("DELETE FROM asset_status WHERE asset_id = ?", [asset_id])
        conn.execute(
            """
            INSERT INTO asset_status
                (asset_id, last_forecast_date, last_run_status, last_run_at,
                 active_bundle_version, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [asset_id, "2024-06-28", status,
             datetime.now(timezone.utc).isoformat(),
             "2026-03-27_prod_abc1234",
             datetime.now(timezone.utc).isoformat()],
        )

    def test_empty_returns_empty(self, db):
        assert get_asset_status_all(db) == []

    def test_returns_all_assets(self, db):
        for a in ["bitcoin", "gold", "us_equities"]:
            self._seed_status(db, a)
        result = get_asset_status_all(db)
        assert len(result) == 3

    def test_ordered_by_asset_id(self, db):
        for a in ["us_equities", "bitcoin", "gold"]:
            self._seed_status(db, a)
        result = get_asset_status_all(db)
        ids = [r["asset_id"] for r in result]
        assert ids == sorted(ids)

    def test_contains_expected_keys(self, db):
        self._seed_status(db, "us_equities")
        result = get_asset_status_all(db)
        expected = {
            "asset_id", "last_forecast_date", "last_run_status",
            "last_run_at", "active_bundle_version", "updated_at",
        }
        assert expected.issubset(result[0].keys())


# ---------------------------------------------------------------------------
# ops — get_ops_summary
# ---------------------------------------------------------------------------

class TestGetOpsSummary:
    def _seed(self, db, asset_id, status):
        conn = db.connect()
        conn.execute("DELETE FROM asset_status WHERE asset_id = ?", [asset_id])
        conn.execute(
            """
            INSERT INTO asset_status
                (asset_id, last_forecast_date, last_run_status, last_run_at,
                 active_bundle_version, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [asset_id, "2024-06-28", status,
             datetime.now(timezone.utc).isoformat(),
             "2026-03-27_prod_abc1234",
             datetime.now(timezone.utc).isoformat()],
        )

    def test_empty_db(self, db):
        summary = get_ops_summary(db)
        assert summary["asset_count"] == 0
        assert summary["fresh_count"] == 0
        assert summary["failed_count"] == 0

    def test_counts_correct(self, db):
        self._seed(db, "us_equities", "fresh")
        self._seed(db, "bitcoin", "fresh")
        self._seed(db, "gold", "stale_news")
        self._seed(db, "long_us_treasuries", "failed")
        summary = get_ops_summary(db)
        assert summary["asset_count"] == 4
        assert summary["fresh_count"] == 2
        assert summary["stale_count"] == 1
        assert summary["failed_count"] == 1

    def test_assets_list_in_summary(self, db):
        self._seed(db, "us_equities", "fresh")
        summary = get_ops_summary(db)
        assert len(summary["assets"]) == 1


# ---------------------------------------------------------------------------
# ops — get_recent_runs
# ---------------------------------------------------------------------------

class TestGetRecentRuns:
    def _seed_run(self, db, asset_id, status="fresh"):
        run_id = str(uuid.uuid4())
        conn = db.connect()
        conn.execute(
            """
            INSERT INTO prediction_runs
                (run_id, forecast_date, asset_id, attempt_no, status, started_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [run_id, "2024-06-28", asset_id, 1, status,
             datetime.now(timezone.utc).isoformat()],
        )
        return run_id

    def test_empty_returns_empty(self, db):
        assert get_recent_runs(db) == []

    def test_returns_runs(self, db):
        self._seed_run(db, "us_equities")
        self._seed_run(db, "bitcoin")
        result = get_recent_runs(db)
        assert len(result) == 2

    def test_limit_respected(self, db):
        for _ in range(10):
            self._seed_run(db, "us_equities")
        result = get_recent_runs(db, limit=5)
        assert len(result) == 5

    def test_asset_filter(self, db):
        self._seed_run(db, "us_equities")
        self._seed_run(db, "bitcoin")
        result = get_recent_runs(db, asset_id="bitcoin")
        assert all(r["asset_id"] == "bitcoin" for r in result)

    def test_contains_expected_keys(self, db):
        self._seed_run(db, "us_equities")
        result = get_recent_runs(db)
        expected = {
            "run_id", "forecast_date", "asset_id", "attempt_no",
            "status", "started_at",
        }
        assert expected.issubset(result[0].keys())


# ---------------------------------------------------------------------------
# ops — get_promotion_history
# ---------------------------------------------------------------------------

class TestGetPromotionHistory:
    def _seed_event(self, db, asset_id, status="promoted"):
        event_id = str(uuid.uuid4())
        conn = db.connect()
        conn.execute(
            """
            INSERT INTO promotion_events
                (event_id, asset_id, release_id, bundle_version,
                 status, promoted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [event_id, asset_id, "prod_2026-03-27", "2026-03-27_prod_abc1234",
             status, datetime.now(timezone.utc).isoformat()],
        )

    def test_empty_returns_empty(self, db):
        assert get_promotion_history(db) == []

    def test_returns_events(self, db):
        self._seed_event(db, "us_equities")
        self._seed_event(db, "bitcoin", status="failed")
        result = get_promotion_history(db)
        assert len(result) == 2

    def test_asset_filter(self, db):
        self._seed_event(db, "us_equities")
        self._seed_event(db, "bitcoin")
        result = get_promotion_history(db, asset_id="us_equities")
        assert all(r["asset_id"] == "us_equities" for r in result)


# ---------------------------------------------------------------------------
# ops — get_active_bundles
# ---------------------------------------------------------------------------

class TestGetActiveBundles:
    def _seed_bundle(self, db, asset_id):
        conn = db.connect()
        conn.execute("DELETE FROM active_bundles WHERE asset_id = ?", [asset_id])
        conn.execute(
            """
            INSERT INTO active_bundles
                (asset_id, source_ticker, release_id, bundle_version,
                 model_type, feature_profile, promoted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [asset_id, "^GSPC", "prod_2026-03-27", "2026-03-27_prod_abc1234",
             "xgb", "gkg_v1_features_compact",
             datetime.now(timezone.utc).isoformat()],
        )

    def test_empty_returns_empty(self, db):
        assert get_active_bundles(db) == []

    def test_returns_bundles(self, db):
        self._seed_bundle(db, "us_equities")
        self._seed_bundle(db, "bitcoin")
        result = get_active_bundles(db)
        assert len(result) == 2

    def test_ordered_by_asset_id(self, db):
        for a in ["us_equities", "bitcoin", "gold"]:
            self._seed_bundle(db, a)
        result = get_active_bundles(db)
        ids = [r["asset_id"] for r in result]
        assert ids == sorted(ids)

    def test_contains_expected_keys(self, db):
        self._seed_bundle(db, "us_equities")
        result = get_active_bundles(db)
        expected = {
            "asset_id", "source_ticker", "release_id", "bundle_version",
            "model_type", "feature_profile", "promoted_at",
        }
        assert expected.issubset(result[0].keys())
