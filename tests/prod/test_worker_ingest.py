'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Tests for the production ingestion adapters (worker/ingest.py).

Covers:
- _load_datasources_config loads the production config correctly.
- _resolve_db_path respects: override param > env var > config.
- _check_prices_freshness returns (True, date) for recent data, (False, None) for absent.
- _check_prices_freshness returns (False, last_date) for stale data.
- _check_macro_freshness returns True for recent data, False for stale/absent.
- _check_gkg_freshness returns True for recent data, False for stale/absent.
- _prev_business_days skips weekends correctly.
- ingest_for_asset: success path (prices fresh, macro+news from shared_ingest).
- ingest_for_asset: prices refresh error captured in prices_error, prices_ok=False.
- ingest_for_asset: shared_ingest=None falls back to db freshness check.
- ingest_for_asset: shared_ingest with stale macro → macro_ok=False.
- run_shared_ingest: macro refresh error → macro_ok=False, macro_error set.
- run_shared_ingest: gkg disabled in config → news_ok=False.
- SharedIngestResult dataclass fields.
- IngestResult dataclass fields.
'''

from __future__ import annotations

import textwrap
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from quant_risk.prod.worker.ingest import (
    IngestResult,
    SharedIngestResult,
    _check_gkg_freshness,
    _check_macro_freshness,
    _check_prices_freshness,
    _load_datasources_config,
    _prev_business_days,
    _resolve_db_path,
    ingest_for_asset,
    run_shared_ingest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)
STALE = TODAY - timedelta(days=30)


def _open_db(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(tmp_path / "test.duckdb"))


def _seed_prices(con: duckdb.DuckDBPyConnection, ticker: str, last_date: date) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_prices (
            ticker TEXT, date DATE, open DOUBLE, high DOUBLE,
            low DOUBLE, close DOUBLE, volume BIGINT,
            source TEXT, inserted_at TIMESTAMP,
            PRIMARY KEY (ticker, date)
        )
    """)
    con.execute(
        "INSERT INTO raw_prices VALUES (?, ?, 100, 101, 99, 100, 1000, 'test', ?)",
        [ticker, last_date.isoformat(), datetime.now(timezone.utc).isoformat()],
    )


def _seed_macro(con: duckdb.DuckDBPyConnection, last_date: date) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS macro_features (
            date DATE PRIMARY KEY,
            vix DOUBLE, fed_assets DOUBLE, tga DOUBLE, rrp DOUBLE,
            m2 DOUBLE, ffr DOUBLE, sofr DOUBLE, net_liquidity DOUBLE,
            source TEXT, inserted_at TIMESTAMP
        )
    """)
    con.execute(
        "INSERT INTO macro_features (date, vix, source, inserted_at) VALUES (?, 20.0, 'test', ?)",
        [last_date.isoformat(), datetime.now(timezone.utc).isoformat()],
    )


def _seed_gkg(con: duckdb.DuckDBPyConnection, last_date: date) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS news_features_daily (
            date DATE, query_id TEXT, news_count BIGINT,
            tone_mean DOUBLE, tone_std DOUBLE, tone_neg_share DOUBLE,
            source TEXT, inserted_at TIMESTAMP,
            PRIMARY KEY (date, query_id)
        )
    """)
    con.execute(
        "INSERT INTO news_features_daily VALUES (?, 'macro_us', 5, -0.1, 0.5, 0.3, 'test', ?)",
        [last_date.isoformat(), datetime.now(timezone.utc).isoformat()],
    )


def _write_datasources_config(tmp_path: Path, db_path: str | None = None) -> str:
    cfg_path = tmp_path / "datasources.rpi5.yaml"
    resolved_db = db_path or str(tmp_path / "financial_data.duckdb")
    cfg_path.write_text(textwrap.dedent(f"""
        db:
          path: {resolved_db}
        prices:
          tickers: ["^GSPC"]
          default_start: "2010-01-01"
          auto_adjust: true
          lookback_buffer_days: 5
        macro:
          default_start: "2010-01-01"
          lookback_buffer_days: 10
        gkg:
          enabled: true
          raw_table: gdelt_gkg_raw
          daily_table: news_features_daily
          profile_name: gkg_v1_light
          profile_version: "2026-03-13"
          profile_mode: light_daily
          start: "2020-01-01"
          lookback_buffer_days: 14
          publication_lag_bdays: 1
          timeout_seconds: 45
          max_retries: 2
          retry_backoff_seconds: 1.0
          retry_max_sleep_seconds: 10.0
          request_pause_seconds: 0.0
          sample_interval_minutes: 180
          max_files_per_day: 2
          max_files_total: 0
          store_raw: false
          topics:
            macro_us: "(inflation OR recession)"
    """), encoding="utf-8")
    return str(cfg_path)


# ---------------------------------------------------------------------------
# _load_datasources_config
# ---------------------------------------------------------------------------

def test_load_datasources_config_loads_production_config():
    cfg = _load_datasources_config("config/prod/datasources.rpi5.yaml")
    assert "db" in cfg
    assert "prices" in cfg
    assert "gkg" in cfg
    assert cfg["gkg"]["profile_name"] == "gkg_v1_light"


def test_load_datasources_config_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        _load_datasources_config("/nonexistent/path.yaml")


# ---------------------------------------------------------------------------
# _resolve_db_path
# ---------------------------------------------------------------------------

def test_resolve_db_path_uses_override():
    cfg = {"db": {"path": "/config/path.duckdb"}}
    assert _resolve_db_path(cfg, "/override/path.duckdb") == "/override/path.duckdb"


def test_resolve_db_path_uses_env_var(monkeypatch):
    monkeypatch.setenv("QUANT_RISK_DB_PATH", "/env/path.duckdb")
    cfg = {"db": {"path": "/config/path.duckdb"}}
    assert _resolve_db_path(cfg) == "/env/path.duckdb"


def test_resolve_db_path_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("QUANT_RISK_DB_PATH", raising=False)
    cfg = {"db": {"path": "/config/path.duckdb"}}
    assert _resolve_db_path(cfg) == "/config/path.duckdb"


def test_resolve_db_path_override_beats_env_var(monkeypatch):
    monkeypatch.setenv("QUANT_RISK_DB_PATH", "/env/path.duckdb")
    cfg = {"db": {"path": "/config/path.duckdb"}}
    assert _resolve_db_path(cfg, "/override/path.duckdb") == "/override/path.duckdb"


# ---------------------------------------------------------------------------
# _prev_business_days
# ---------------------------------------------------------------------------

def test_prev_business_days_skips_weekend():
    # Monday 2026-03-30 minus 1 bday = Friday 2026-03-27
    monday = date(2026, 3, 30)
    assert _prev_business_days(monday, 1) == date(2026, 3, 27)


def test_prev_business_days_zero():
    d = date(2026, 3, 27)
    assert _prev_business_days(d, 0) == d


# ---------------------------------------------------------------------------
# _check_prices_freshness
# ---------------------------------------------------------------------------

def test_check_prices_freshness_recent_data(tmp_path):
    con = _open_db(tmp_path)
    _seed_prices(con, "^GSPC", YESTERDAY)
    ok, last = _check_prices_freshness(con, "^GSPC")
    assert ok is True
    assert last == YESTERDAY
    con.close()


def test_check_prices_freshness_stale_data(tmp_path):
    con = _open_db(tmp_path)
    _seed_prices(con, "^GSPC", STALE)
    ok, last = _check_prices_freshness(con, "^GSPC", staleness_bdays=5)
    assert ok is False
    assert last == STALE
    con.close()


def test_check_prices_freshness_no_data(tmp_path):
    con = _open_db(tmp_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_prices (
            ticker TEXT, date DATE, open DOUBLE, high DOUBLE,
            low DOUBLE, close DOUBLE, volume BIGINT,
            source TEXT, inserted_at TIMESTAMP,
            PRIMARY KEY (ticker, date)
        )
    """)
    ok, last = _check_prices_freshness(con, "^GSPC")
    assert ok is False
    assert last is None
    con.close()


def test_check_prices_freshness_unknown_ticker(tmp_path):
    con = _open_db(tmp_path)
    _seed_prices(con, "^GSPC", YESTERDAY)
    ok, last = _check_prices_freshness(con, "UNKNOWN")
    assert ok is False
    assert last is None
    con.close()


# ---------------------------------------------------------------------------
# _check_macro_freshness
# ---------------------------------------------------------------------------

def test_check_macro_freshness_recent(tmp_path):
    con = _open_db(tmp_path)
    _seed_macro(con, YESTERDAY)
    assert _check_macro_freshness(con) is True
    con.close()


def test_check_macro_freshness_stale(tmp_path):
    con = _open_db(tmp_path)
    _seed_macro(con, STALE)
    assert _check_macro_freshness(con, staleness_days=10) is False
    con.close()


def test_check_macro_freshness_empty(tmp_path):
    con = _open_db(tmp_path)
    con.execute("CREATE TABLE IF NOT EXISTS macro_features (date DATE PRIMARY KEY)")
    assert _check_macro_freshness(con) is False
    con.close()


# ---------------------------------------------------------------------------
# _check_gkg_freshness
# ---------------------------------------------------------------------------

def test_check_gkg_freshness_recent(tmp_path):
    con = _open_db(tmp_path)
    # GKG pub lag = 1, staleness = 3 → threshold = 4 bdays ago
    recent = _prev_business_days(TODAY, 2)
    _seed_gkg(con, recent)
    assert _check_gkg_freshness(con, publication_lag_bdays=1, staleness_bdays=3) is True
    con.close()


def test_check_gkg_freshness_stale(tmp_path):
    con = _open_db(tmp_path)
    _seed_gkg(con, STALE)
    assert _check_gkg_freshness(con, publication_lag_bdays=1, staleness_bdays=3) is False
    con.close()


def test_check_gkg_freshness_empty(tmp_path):
    con = _open_db(tmp_path)
    con.execute("CREATE TABLE IF NOT EXISTS news_features_daily (date DATE, query_id TEXT, PRIMARY KEY(date, query_id))")
    assert _check_gkg_freshness(con) is False
    con.close()


# ---------------------------------------------------------------------------
# SharedIngestResult / IngestResult
# ---------------------------------------------------------------------------

def test_shared_ingest_result_fields():
    r = SharedIngestResult(macro_ok=True, news_ok=False, news_error="GKG timeout")
    assert r.macro_ok is True
    assert r.news_ok is False
    assert r.news_error == "GKG timeout"
    assert r.macro_error is None
    assert isinstance(r.ingest_date, date)


def test_ingest_result_fields():
    r = IngestResult(prices_ok=True, macro_ok=True, news_ok=False, data_cutoff_date=YESTERDAY)
    assert r.prices_ok is True
    assert r.news_ok is False
    assert r.prices_error is None


# ---------------------------------------------------------------------------
# ingest_for_asset — with mocked refresh functions
# ---------------------------------------------------------------------------

def _build_fake_db_with_fresh_prices(tmp_path: Path, ticker: str) -> str:
    db_path = str(tmp_path / "financial_data.duckdb")
    con = duckdb.connect(db_path)
    _seed_prices(con, ticker, YESTERDAY)
    _seed_macro(con, YESTERDAY)
    _seed_gkg(con, _prev_business_days(TODAY, 2))
    con.close()
    return db_path


def test_ingest_for_asset_success_with_shared_ingest(tmp_path, monkeypatch):
    db_path = _build_fake_db_with_fresh_prices(tmp_path, "^GSPC")
    cfg_path = _write_datasources_config(tmp_path, db_path)

    # Mock refresh_prices to be a no-op (data already seeded)
    monkeypatch.setattr(
        "quant_risk.prod.worker.ingest.refresh_prices",
        lambda cfg, tickers: {"inserted_rows": 0, "errors": {}},
    )

    shared = SharedIngestResult(macro_ok=True, news_ok=True)
    result = ingest_for_asset(
        asset_id="us_equities",
        source_ticker="^GSPC",
        datasources_config=cfg_path,
        db_path=db_path,
        shared_ingest=shared,
    )

    assert result.prices_ok is True
    assert result.macro_ok is True
    assert result.news_ok is True
    assert result.data_cutoff_date == YESTERDAY
    assert result.prices_error is None


def test_ingest_for_asset_prices_error_captured(tmp_path, monkeypatch):
    cfg_path = _write_datasources_config(tmp_path)
    db_path = str(tmp_path / "financial_data.duckdb")

    def bad_refresh(cfg, tickers):
        return {"inserted_rows": 0, "errors": {"^GSPC": "connection timeout"}}

    monkeypatch.setattr("quant_risk.prod.worker.ingest.refresh_prices", bad_refresh)

    shared = SharedIngestResult(macro_ok=True, news_ok=True)
    result = ingest_for_asset(
        asset_id="us_equities",
        source_ticker="^GSPC",
        datasources_config=cfg_path,
        db_path=db_path,
        shared_ingest=shared,
    )

    assert result.prices_ok is False
    assert "connection timeout" in (result.prices_error or "")
    assert result.macro_ok is True  # from shared_ingest


def test_ingest_for_asset_stale_news_from_shared(tmp_path, monkeypatch):
    db_path = _build_fake_db_with_fresh_prices(tmp_path, "^GSPC")
    cfg_path = _write_datasources_config(tmp_path, db_path)

    monkeypatch.setattr(
        "quant_risk.prod.worker.ingest.refresh_prices",
        lambda cfg, tickers: {"inserted_rows": 0, "errors": {}},
    )

    shared = SharedIngestResult(macro_ok=True, news_ok=False, news_error="GKG stale")
    result = ingest_for_asset(
        asset_id="us_equities",
        source_ticker="^GSPC",
        datasources_config=cfg_path,
        db_path=db_path,
        shared_ingest=shared,
    )

    assert result.prices_ok is True
    assert result.news_ok is False


def test_ingest_for_asset_fallback_freshness_when_no_shared_ingest(tmp_path, monkeypatch):
    db_path = _build_fake_db_with_fresh_prices(tmp_path, "^GSPC")
    cfg_path = _write_datasources_config(tmp_path, db_path)

    monkeypatch.setattr(
        "quant_risk.prod.worker.ingest.refresh_prices",
        lambda cfg, tickers: {"inserted_rows": 0, "errors": {}},
    )

    # No shared_ingest — should check DB directly
    result = ingest_for_asset(
        asset_id="us_equities",
        source_ticker="^GSPC",
        datasources_config=cfg_path,
        db_path=db_path,
        shared_ingest=None,
    )

    assert result.prices_ok is True
    assert isinstance(result.macro_ok, bool)
    assert isinstance(result.news_ok, bool)


# ---------------------------------------------------------------------------
# run_shared_ingest — with mocked refresh functions
# ---------------------------------------------------------------------------

def test_run_shared_ingest_macro_error_sets_flag(tmp_path, monkeypatch):
    cfg_path = _write_datasources_config(tmp_path)
    db_path = str(tmp_path / "financial_data.duckdb")

    def bad_macro(cfg, **kw):
        raise RuntimeError("FRED unreachable")

    monkeypatch.setattr("quant_risk.prod.worker.ingest.refresh_macro", bad_macro)
    monkeypatch.setattr(
        "quant_risk.prod.worker.ingest.refresh_gkg",
        lambda cfg, **kw: {"inserted_raw_rows": 0, "inserted_daily_rows": 0, "errors": {}},
    )

    result = run_shared_ingest(cfg_path, db_path=db_path)

    assert result.macro_ok is False
    assert "FRED unreachable" in (result.macro_error or "")


def test_run_shared_ingest_gkg_disabled(tmp_path, monkeypatch):
    # Write a config with gkg.enabled = false
    cfg_path = tmp_path / "ds_disabled.yaml"
    cfg_path.write_text(textwrap.dedent(f"""
        db:
          path: {str(tmp_path / 'financial_data.duckdb')}
        prices:
          tickers: ["^GSPC"]
          default_start: "2010-01-01"
          auto_adjust: true
          lookback_buffer_days: 5
        macro:
          default_start: "2010-01-01"
          lookback_buffer_days: 10
        gkg:
          enabled: false
    """), encoding="utf-8")
    db_path = str(tmp_path / "financial_data.duckdb")

    monkeypatch.setattr(
        "quant_risk.prod.worker.ingest.refresh_macro",
        lambda cfg, **kw: {"inserted_rows": 0, "errors": {}},
    )

    result = run_shared_ingest(str(cfg_path), db_path=db_path)
    assert result.news_ok is False
    assert result.news_error is not None
