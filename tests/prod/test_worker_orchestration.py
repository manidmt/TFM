'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Tests for the production worker entrypoint and run orchestration.

Covers:
- RunContext / RunResult construction.
- start_run inserts a row in prediction_runs with attempt_no=1.
- start_run auto-increments attempt_no on subsequent calls.
- finish_run updates the prediction_runs row with final status and finished_at.
- update_asset_status upserts asset_status correctly.
- update_asset_status is idempotent (second call overwrites the first).
- make_failed_result returns a RunResult with status=failed.
- run_asset_batch partial success: success path with injected stubs.
- run_asset_batch failure path: exception in ingest captured, status=failed.
- run_asset_batch dry_run: skips I/O, returns fresh status.
- run_daily_batch processes all assets independently (partial success).
- run_daily_batch retries a failed asset once.
- _resolve_forecast_date defaults to yesterday.
- _determine_run_status maps IngestResult flags to RunStatus correctly.
'''

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from quant_risk.prod.schemas import PredictedClass, RunStatus
from quant_risk.prod.serving.duckdb import ServingDB
from quant_risk.prod.worker.ingest import IngestResult
from quant_risk.prod.worker.inference import PredictionOutput
from quant_risk.prod.worker.main import (
    WorkerConfig,
    _determine_run_status,
    _resolve_forecast_date,
    run_asset_batch,
    run_daily_batch,
)
from quant_risk.prod.worker.runs import (
    RunContext,
    RunResult,
    finish_run,
    make_failed_result,
    start_run,
    update_asset_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)


def _make_db(tmp_path: Path) -> ServingDB:
    db = ServingDB(tmp_path / "serving.duckdb")
    db.create_schema()
    return db


def _make_asset_info(asset_id: str = "us_equities", ticker: str = "^GSPC"):
    from quant_risk.prod.assets import AssetInfo
    return AssetInfo(
        asset_id=asset_id,
        display_name="US Equities",
        source_ticker=ticker,
        category="Equity",
        short_description="Test asset",
    )


def _ok_ingest_result(cutoff: date | None = None) -> IngestResult:
    return IngestResult(
        prices_ok=True,
        macro_ok=True,
        news_ok=True,
        data_cutoff_date=cutoff or YESTERDAY,
    )


def _make_prediction(bundle: str = "2026-03-27_prod_abc1234") -> PredictionOutput:
    return PredictionOutput(
        predicted_class=PredictedClass.high,
        p_low=0.1,
        p_medium=0.2,
        p_high=0.7,
        bundle_version=bundle,
    )


# ---------------------------------------------------------------------------
# RunContext / RunResult basics
# ---------------------------------------------------------------------------

def test_run_context_fields():
    ctx = RunContext(
        run_id=str(uuid.uuid4()),
        asset_id="us_equities",
        forecast_date=YESTERDAY,
        attempt_no=1,
        started_at=datetime.now(timezone.utc),
    )
    assert ctx.asset_id == "us_equities"
    assert ctx.attempt_no == 1


def test_run_result_is_successful_true():
    r = RunResult(status=RunStatus.fresh, bundle_version="v1")
    assert r.is_successful() is True


def test_run_result_is_successful_false():
    r = RunResult(status=RunStatus.failed)
    assert r.is_successful() is False


def test_make_failed_result():
    r = make_failed_result("Something broke", error_code="MY_ERROR")
    assert r.status == RunStatus.failed
    assert r.error_code == "MY_ERROR"
    assert "Something broke" in r.error_message
    assert r.finished_at is not None


# ---------------------------------------------------------------------------
# start_run
# ---------------------------------------------------------------------------

def test_start_run_inserts_row(tmp_path):
    db = _make_db(tmp_path)
    ctx = start_run(db, "us_equities", YESTERDAY)
    rows = db.connect().execute(
        "SELECT run_id, attempt_no, status FROM prediction_runs WHERE asset_id = ?",
        ["us_equities"],
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == ctx.run_id
    assert rows[0][1] == 1
    assert rows[0][2] == RunStatus.failed.value  # default until finish_run
    db.close()


def test_start_run_increments_attempt_no(tmp_path):
    db = _make_db(tmp_path)
    ctx1 = start_run(db, "us_equities", YESTERDAY)
    ctx2 = start_run(db, "us_equities", YESTERDAY)
    assert ctx1.attempt_no == 1
    assert ctx2.attempt_no == 2
    db.close()


def test_start_run_different_assets_independent(tmp_path):
    db = _make_db(tmp_path)
    ctx_us = start_run(db, "us_equities", YESTERDAY)
    ctx_btc = start_run(db, "bitcoin", YESTERDAY)
    assert ctx_us.attempt_no == 1
    assert ctx_btc.attempt_no == 1
    db.close()


def test_start_run_different_dates_independent(tmp_path):
    db = _make_db(tmp_path)
    ctx1 = start_run(db, "us_equities", YESTERDAY)
    ctx2 = start_run(db, "us_equities", YESTERDAY - timedelta(days=1))
    assert ctx1.attempt_no == 1
    assert ctx2.attempt_no == 1
    db.close()


# ---------------------------------------------------------------------------
# finish_run
# ---------------------------------------------------------------------------

def test_finish_run_updates_status(tmp_path):
    db = _make_db(tmp_path)
    ctx = start_run(db, "us_equities", YESTERDAY)
    result = RunResult(status=RunStatus.fresh, bundle_version="v1")
    finish_run(db, ctx, result)
    row = db.connect().execute(
        "SELECT status, bundle_version FROM prediction_runs WHERE run_id = ?",
        [ctx.run_id],
    ).fetchone()
    assert row[0] == RunStatus.fresh.value
    assert row[1] == "v1"
    db.close()


def test_finish_run_records_error_message(tmp_path):
    db = _make_db(tmp_path)
    ctx = start_run(db, "us_equities", YESTERDAY)
    result = RunResult(
        status=RunStatus.failed,
        error_code="INGEST_ERROR",
        error_message="prices download failed",
    )
    finish_run(db, ctx, result)
    row = db.connect().execute(
        "SELECT error_code, error_message FROM prediction_runs WHERE run_id = ?",
        [ctx.run_id],
    ).fetchone()
    assert row[0] == "INGEST_ERROR"
    assert "prices download failed" in row[1]
    db.close()


# ---------------------------------------------------------------------------
# update_asset_status
# ---------------------------------------------------------------------------

def test_update_asset_status_upserts_row(tmp_path):
    db = _make_db(tmp_path)
    ctx = start_run(db, "us_equities", YESTERDAY)
    result = RunResult(status=RunStatus.fresh, bundle_version="v1")
    update_asset_status(db, ctx, result)
    row = db.connect().execute(
        "SELECT last_run_status, active_bundle_version FROM asset_status WHERE asset_id = ?",
        ["us_equities"],
    ).fetchone()
    assert row[0] == RunStatus.fresh.value
    assert row[1] == "v1"
    db.close()


def test_update_asset_status_is_idempotent(tmp_path):
    db = _make_db(tmp_path)
    ctx = start_run(db, "us_equities", YESTERDAY)
    result1 = RunResult(status=RunStatus.stale_news, bundle_version="v1")
    update_asset_status(db, ctx, result1)
    result2 = RunResult(status=RunStatus.fresh, bundle_version="v2")
    update_asset_status(db, ctx, result2)
    rows = db.connect().execute(
        "SELECT COUNT(*) FROM asset_status WHERE asset_id = ?",
        ["us_equities"],
    ).fetchone()
    assert rows[0] == 1  # only one row, not two
    row = db.connect().execute(
        "SELECT last_run_status FROM asset_status WHERE asset_id = ?",
        ["us_equities"],
    ).fetchone()
    assert row[0] == RunStatus.fresh.value  # latest wins
    db.close()


# ---------------------------------------------------------------------------
# _determine_run_status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prices,macro,news,expected", [
    (True,  True,  True,  RunStatus.fresh),
    (True,  True,  False, RunStatus.stale_news),
    (True,  False, True,  RunStatus.stale_data),
    (False, True,  True,  RunStatus.stale_data),
    (False, False, False, RunStatus.stale_data),
])
def test_determine_run_status(prices, macro, news, expected):
    ir = IngestResult(prices_ok=prices, macro_ok=macro, news_ok=news, data_cutoff_date=YESTERDAY)
    assert _determine_run_status(ir) == expected


# ---------------------------------------------------------------------------
# _resolve_forecast_date
# ---------------------------------------------------------------------------

def test_resolve_forecast_date_defaults_to_yesterday():
    cfg = WorkerConfig()
    assert _resolve_forecast_date(cfg) == date.today() - timedelta(days=1)


def test_resolve_forecast_date_respects_override():
    fixed = date(2026, 1, 15)
    cfg = WorkerConfig(forecast_date=fixed)
    assert _resolve_forecast_date(cfg) == fixed


# ---------------------------------------------------------------------------
# run_asset_batch with injected stubs
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, **overrides) -> WorkerConfig:
    return WorkerConfig(
        assets_config="config/prod/assets.yaml",
        serving_db_path=str(tmp_path / "serving.duckdb"),
        forecast_date=YESTERDAY,
        max_retries=0,
        **overrides,
    )


def test_run_asset_batch_success_path(tmp_path):
    db = _make_db(tmp_path)
    asset = _make_asset_info()
    config = _make_config(tmp_path)
    prediction = _make_prediction()
    ingest_result = _ok_ingest_result()

    result = run_asset_batch(
        asset=asset,
        config=config,
        db=db,
        forecast_date=YESTERDAY,
        _ingest_fn=lambda **kw: ingest_result,
        _features_fn=lambda **kw: MagicMock(features=MagicMock(), feature_count=10, feature_date=YESTERDAY, missing_columns=[]),
        _inference_fn=lambda **kw: prediction,
        _persist_fn=lambda **kw: None,
    )

    assert result.status == RunStatus.fresh
    assert result.bundle_version == prediction.bundle_version
    db.close()


def test_run_asset_batch_stale_news_status(tmp_path):
    db = _make_db(tmp_path)
    asset = _make_asset_info()
    config = _make_config(tmp_path)
    stale_ingest = IngestResult(
        prices_ok=True, macro_ok=True, news_ok=False, data_cutoff_date=YESTERDAY
    )
    prediction = _make_prediction()

    result = run_asset_batch(
        asset=asset,
        config=config,
        db=db,
        forecast_date=YESTERDAY,
        _ingest_fn=lambda **kw: stale_ingest,
        _features_fn=lambda **kw: MagicMock(features=MagicMock(), feature_count=10, feature_date=YESTERDAY, missing_columns=[]),
        _inference_fn=lambda **kw: prediction,
        _persist_fn=lambda **kw: None,
    )

    assert result.status == RunStatus.stale_news
    db.close()


def test_run_asset_batch_failure_captured(tmp_path):
    db = _make_db(tmp_path)
    asset = _make_asset_info()
    config = _make_config(tmp_path)

    def boom(**kw):
        raise RuntimeError("network timeout")

    result = run_asset_batch(
        asset=asset,
        config=config,
        db=db,
        forecast_date=YESTERDAY,
        _ingest_fn=boom,
        _features_fn=lambda **kw: None,
        _inference_fn=lambda **kw: None,
        _persist_fn=lambda **kw: None,
    )

    assert result.status == RunStatus.failed
    assert "network timeout" in result.error_message
    assert result.error_code == "RuntimeError"
    db.close()


def test_run_asset_batch_dry_run(tmp_path):
    db = _make_db(tmp_path)
    asset = _make_asset_info()
    config = _make_config(tmp_path, dry_run=True)

    call_log: list[str] = []

    def track(name):
        def fn(**kw):
            call_log.append(name)
            return MagicMock()
        return fn

    result = run_asset_batch(
        asset=asset,
        config=config,
        db=db,
        forecast_date=YESTERDAY,
        _ingest_fn=track("ingest"),
        _features_fn=track("features"),
        _inference_fn=track("inference"),
        _persist_fn=track("persist"),
    )

    assert result.status == RunStatus.fresh
    assert call_log == []   # no I/O steps were called

    # dry_run must be fully side-effect-free: prediction_runs and asset_status
    # must remain empty (the finally block must NOT have fired).
    runs = db.connect().execute("SELECT COUNT(*) FROM prediction_runs").fetchone()[0]
    status_rows = db.connect().execute("SELECT COUNT(*) FROM asset_status").fetchone()[0]
    assert runs == 0, "dry_run wrote rows to prediction_runs"
    assert status_rows == 0, "dry_run wrote rows to asset_status"
    db.close()


def test_run_asset_batch_writes_to_prediction_runs(tmp_path):
    db = _make_db(tmp_path)
    asset = _make_asset_info()
    config = _make_config(tmp_path)
    prediction = _make_prediction()

    run_asset_batch(
        asset=asset,
        config=config,
        db=db,
        forecast_date=YESTERDAY,
        _ingest_fn=lambda **kw: _ok_ingest_result(),
        _features_fn=lambda **kw: MagicMock(features=MagicMock(), feature_count=5, feature_date=YESTERDAY, missing_columns=[]),
        _inference_fn=lambda **kw: prediction,
        _persist_fn=lambda **kw: None,
    )

    rows = db.connect().execute(
        "SELECT status, bundle_version FROM prediction_runs WHERE asset_id = ?",
        ["us_equities"],
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == RunStatus.fresh.value
    assert rows[0][1] == prediction.bundle_version
    db.close()


# ---------------------------------------------------------------------------
# run_daily_batch
# ---------------------------------------------------------------------------

def _make_full_config(tmp_path: Path) -> WorkerConfig:
    return WorkerConfig(
        assets_config="config/prod/assets.yaml",
        serving_db_path=str(tmp_path / "serving.duckdb"),
        forecast_date=YESTERDAY,
        max_retries=0,
    )


def _mock_shared_ingest(monkeypatch) -> None:
    from quant_risk.prod.worker.ingest import SharedIngestResult
    monkeypatch.setattr(
        "quant_risk.prod.worker.main.run_shared_ingest",
        lambda **kw: SharedIngestResult(macro_ok=True, news_ok=True),
    )


def test_run_daily_batch_processes_all_six_assets(tmp_path, monkeypatch):
    prediction = _make_prediction()

    _mock_shared_ingest(monkeypatch)
    monkeypatch.setattr(
        "quant_risk.prod.worker.main.ingest_for_asset",
        lambda **kw: _ok_ingest_result(),
    )
    monkeypatch.setattr(
        "quant_risk.prod.worker.main.build_features_for_asset",
        lambda **kw: MagicMock(features=MagicMock(), feature_count=5, feature_date=YESTERDAY, missing_columns=[]),
    )
    monkeypatch.setattr(
        "quant_risk.prod.worker.main.run_inference_for_asset",
        lambda **kw: prediction,
    )
    monkeypatch.setattr(
        "quant_risk.prod.worker.main.persist_prediction",
        lambda **kw: None,
    )

    config = _make_full_config(tmp_path)
    statuses = run_daily_batch(config)

    assert len(statuses) == 6
    assert all(s == RunStatus.fresh for s in statuses.values())


def test_run_daily_batch_partial_success(tmp_path, monkeypatch):
    """One asset fails; the other five must still complete."""
    call_counts: dict[str, int] = {}

    def selective_ingest(**kw):
        aid = kw["asset_id"]
        call_counts[aid] = call_counts.get(aid, 0) + 1
        if aid == "bitcoin":
            raise RuntimeError("bitcoin API down")
        return _ok_ingest_result()

    prediction = _make_prediction()
    _mock_shared_ingest(monkeypatch)
    monkeypatch.setattr("quant_risk.prod.worker.main.ingest_for_asset", selective_ingest)
    monkeypatch.setattr(
        "quant_risk.prod.worker.main.build_features_for_asset",
        lambda **kw: MagicMock(features=MagicMock(), feature_count=5, feature_date=YESTERDAY, missing_columns=[]),
    )
    monkeypatch.setattr(
        "quant_risk.prod.worker.main.run_inference_for_asset",
        lambda **kw: prediction,
    )
    monkeypatch.setattr("quant_risk.prod.worker.main.persist_prediction", lambda **kw: None)

    config = _make_full_config(tmp_path)
    statuses = run_daily_batch(config)

    assert statuses["bitcoin"] == RunStatus.failed
    assert statuses["us_equities"] == RunStatus.fresh
    assert sum(1 for s in statuses.values() if s != RunStatus.failed) == 5


def test_run_daily_batch_retry_on_failure(tmp_path, monkeypatch):
    """A failed asset should be retried once when max_retries=1."""
    call_counts: dict[str, int] = {"bitcoin": 0}

    def flaky_ingest(**kw):
        if kw["asset_id"] == "bitcoin":
            call_counts["bitcoin"] += 1
            if call_counts["bitcoin"] == 1:
                raise RuntimeError("transient error")
            return _ok_ingest_result()
        return _ok_ingest_result()

    prediction = _make_prediction()
    _mock_shared_ingest(monkeypatch)
    monkeypatch.setattr("quant_risk.prod.worker.main.ingest_for_asset", flaky_ingest)
    monkeypatch.setattr(
        "quant_risk.prod.worker.main.build_features_for_asset",
        lambda **kw: MagicMock(features=MagicMock(), feature_count=5, feature_date=YESTERDAY, missing_columns=[]),
    )
    monkeypatch.setattr("quant_risk.prod.worker.main.run_inference_for_asset", lambda **kw: prediction)
    monkeypatch.setattr("quant_risk.prod.worker.main.persist_prediction", lambda **kw: None)

    config = WorkerConfig(
        assets_config="config/prod/assets.yaml",
        serving_db_path=str(tmp_path / "serving.duckdb"),
        forecast_date=YESTERDAY,
        max_retries=1,
    )
    statuses = run_daily_batch(config)

    assert statuses["bitcoin"] == RunStatus.fresh
    assert call_counts["bitcoin"] == 2  # first attempt + one retry
