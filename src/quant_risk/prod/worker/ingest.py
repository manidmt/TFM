'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Production ingestion adapters for the RPi5 worker.

Thin wrappers over src/quant_risk/data/fetcher.py, macro.py, and gkg.py that:
- consume config/prod/datasources.rpi5.yaml
- write to financial_data.duckdb
- return typed IngestResult / SharedIngestResult per run

Design (rpi5.md §10.3):
  - Macro and GKG refresh is SHARED across all assets (run once per batch via
    run_shared_ingest).  Calling it 6 times would be idempotent but wasteful.
  - Price refresh is PER-ASSET (ingest_for_asset) because each ticker is
    independent and can fail without affecting the others.
  - Per-asset IngestResult is derived from per-ticker price freshness +
    the shared macro/GKG result.

Reference: rpi5.md §10.3, §26.2, §26.3
'''

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb

from quant_risk.config import load_yaml
from quant_risk.data.fetcher import PriceIngestConfig, refresh_prices
from quant_risk.data.gkg import GkgIngestConfig, refresh_gkg
from quant_risk.data.macro import MacroIngestConfig, refresh_macro

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SharedIngestResult:
    """Outcome of the shared (non-asset-specific) ingestion phase.

    Attributes
    ----------
    macro_ok:
        True if macro_features is fresh enough for inference.
    news_ok:
        True if news_features_daily is fresh enough for inference.
    macro_error:
        Error message if macro refresh failed, else None.
    news_error:
        Error message if GKG refresh failed, else None.
    ingest_date:
        The date on which this shared ingest ran (today).
    """

    macro_ok: bool
    news_ok: bool
    macro_error: Optional[str] = None
    news_error: Optional[str] = None
    ingest_date: date = field(default_factory=date.today)


@dataclass
class IngestResult:
    """Outcome of a production ingestion run for a single asset.

    Attributes
    ----------
    prices_ok:
        True if the latest D-1 price bar was successfully ingested.
    macro_ok:
        True if macro features are within the acceptable staleness window.
    news_ok:
        True if GKG/news data is within the acceptable staleness window.
    data_cutoff_date:
        The most recent market date for which price data is available.
    prices_error:
        Error detail if price ingestion failed.
    """

    prices_ok: bool
    macro_ok: bool
    news_ok: bool
    data_cutoff_date: date
    prices_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_datasources_config(datasources_config: str) -> dict:
    """Load datasources.rpi5.yaml and return the parsed dict."""
    return load_yaml(datasources_config)


def _resolve_db_path(cfg: dict, override: str | None = None) -> str:
    """Resolve financial_data.duckdb path.

    Priority:
    1. ``override`` parameter (explicit caller override)
    2. ``QUANT_RISK_DB_PATH`` environment variable
    3. ``db.path`` from the datasources YAML config
    """
    if override:
        return override
    env = os.environ.get("QUANT_RISK_DB_PATH")
    if env:
        return env
    return str(cfg["db"]["path"])


# ---------------------------------------------------------------------------
# Freshness checks
# ---------------------------------------------------------------------------

# Maximum age of macro data before we flag it as stale (days).
_MACRO_STALENESS_DAYS = 10
# Maximum age of GKG daily data before we flag it as stale (business days).
_GKG_STALENESS_BDAYS = 3
# Maximum age of price data before we flag it as stale (business days).
_PRICE_STALENESS_BDAYS = 5


def _prev_business_days(ref: date, n: int) -> date:
    """Return the date *n* business days before *ref*."""
    d = ref
    for _ in range(n):
        d -= timedelta(days=1)
        while d.weekday() >= 5:  # Sat=5, Sun=6
            d -= timedelta(days=1)
    return d


def _check_prices_freshness(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    table: str = "raw_prices",
    staleness_bdays: int = _PRICE_STALENESS_BDAYS,
) -> tuple[bool, date | None]:
    """Return (is_fresh, last_date) for a ticker's price data."""
    row = con.execute(
        f"SELECT MAX(date) FROM {table} WHERE ticker = ?",
        [ticker],
    ).fetchone()
    last_date: date | None = row[0] if row and row[0] is not None else None
    if last_date is None:
        return False, None
    threshold = _prev_business_days(date.today(), staleness_bdays)
    return last_date >= threshold, last_date


def _check_macro_freshness(
    con: duckdb.DuckDBPyConnection,
    table: str = "macro_features",
    staleness_days: int = _MACRO_STALENESS_DAYS,
) -> bool:
    """Return True if macro data is not excessively stale."""
    row = con.execute(f"SELECT MAX(date) FROM {table}").fetchone()
    last_date: date | None = row[0] if row and row[0] is not None else None
    if last_date is None:
        return False
    threshold = date.today() - timedelta(days=staleness_days)
    return last_date >= threshold


def _check_gkg_freshness(
    con: duckdb.DuckDBPyConnection,
    table: str = "news_features_daily",
    publication_lag_bdays: int = 1,
    staleness_bdays: int = _GKG_STALENESS_BDAYS,
) -> bool:
    """Return True if GKG daily news data is not excessively stale.

    GKG has a publication lag of ``publication_lag_bdays`` business days
    (data from day D appears on D+1).  We therefore expect GKG to be
    available up to ``publication_lag_bdays`` + ``staleness_bdays`` business
    days before today.
    """
    row = con.execute(f"SELECT MAX(date) FROM {table}").fetchone()
    last_date: date | None = row[0] if row and row[0] is not None else None
    if last_date is None:
        return False
    expected_lag = publication_lag_bdays + staleness_bdays
    threshold = _prev_business_days(date.today(), expected_lag)
    return last_date >= threshold


# ---------------------------------------------------------------------------
# Shared ingest (macro + GKG — run once per batch)
# ---------------------------------------------------------------------------

def run_shared_ingest(
    datasources_config: str,
    db_path: str | None = None,
) -> SharedIngestResult:
    """Refresh macro and GKG/news data — shared across all production assets.

    This function should be called ONCE per daily batch before the per-asset
    loop.  It is idempotent: calling it multiple times on the same day is safe
    (all upserts are ON CONFLICT … DO UPDATE).

    Parameters
    ----------
    datasources_config:
        Path to config/prod/datasources.rpi5.yaml.
    db_path:
        Override for financial_data.duckdb path.  Falls back to
        QUANT_RISK_DB_PATH env var, then to datasources config.
    """
    cfg = _load_datasources_config(datasources_config)
    resolved_db = _resolve_db_path(cfg, db_path)
    Path(resolved_db).parent.mkdir(parents=True, exist_ok=True)

    macro_ok = False
    news_ok = False
    macro_error: str | None = None
    news_error: str | None = None

    # --- Macro ---
    try:
        macro_cfg = MacroIngestConfig(
            db_path=resolved_db,
            default_start=str(cfg["macro"].get("default_start", "2010-01-01")),
            lookback_buffer_days=int(cfg["macro"].get("lookback_buffer_days", 10)),
        )
        res = refresh_macro(macro_cfg)
        if res.get("errors"):
            macro_error = str(res["errors"])
            logger.warning("Macro ingest had errors: %s", macro_error)
        # Check freshness regardless of errors (we may have prior data)
        con = duckdb.connect(resolved_db, read_only=True)
        try:
            macro_ok = _check_macro_freshness(con)
        finally:
            con.close()
    except Exception as exc:
        macro_error = str(exc)
        logger.exception("Macro ingest failed: %s", exc)

    # --- GKG / news ---
    gkg_cfg_raw = cfg.get("gkg", {})
    if not bool(gkg_cfg_raw.get("enabled", True)):
        logger.info("GKG ingest disabled in config — marking news as stale.")
        news_ok = False
        news_error = "GKG disabled in datasources config."
    else:
        try:
            topics_raw = gkg_cfg_raw.get("topics", {})
            topics = (
                {str(k): str(v) for k, v in topics_raw.items()}
                if isinstance(topics_raw, dict)
                else None
            )
            gkg_cfg = GkgIngestConfig(
                db_path=resolved_db,
                raw_table=str(gkg_cfg_raw.get("raw_table", "gdelt_gkg_raw")),
                daily_table=str(gkg_cfg_raw.get("daily_table", "news_features_daily")),
                profile_name=str(gkg_cfg_raw.get("profile_name", "gkg_v1_light")),
                profile_version=str(gkg_cfg_raw.get("profile_version", "2026-03-13")),
                profile_mode=str(gkg_cfg_raw.get("profile_mode", "light_daily")),
                start=str(gkg_cfg_raw.get("start", "2020-01-01")),
                lookback_buffer_days=int(gkg_cfg_raw.get("lookback_buffer_days", 14)),
                publication_lag_bdays=int(gkg_cfg_raw.get("publication_lag_bdays", 1)),
                timeout_seconds=int(gkg_cfg_raw.get("timeout_seconds", 45)),
                max_retries=int(gkg_cfg_raw.get("max_retries", 8)),
                retry_backoff_seconds=float(gkg_cfg_raw.get("retry_backoff_seconds", 3.0)),
                retry_max_sleep_seconds=float(gkg_cfg_raw.get("retry_max_sleep_seconds", 240.0)),
                request_pause_seconds=float(gkg_cfg_raw.get("request_pause_seconds", 1.0)),
                sample_interval_minutes=int(gkg_cfg_raw.get("sample_interval_minutes", 180)),
                max_files_per_day=int(gkg_cfg_raw.get("max_files_per_day", 8)),
                max_files_total=int(gkg_cfg_raw.get("max_files_total", 0)),
                store_raw=bool(gkg_cfg_raw.get("store_raw", False)),
                topics=topics,
                verbose=False,
            )
            res = refresh_gkg(gkg_cfg)
            if res.get("errors"):
                news_error = str(res["errors"])
                logger.warning("GKG ingest had errors: %s", news_error)
            # Check freshness
            pub_lag = int(gkg_cfg_raw.get("publication_lag_bdays", 1))
            con = duckdb.connect(resolved_db, read_only=True)
            try:
                news_ok = _check_gkg_freshness(
                    con,
                    table=str(gkg_cfg_raw.get("daily_table", "news_features_daily")),
                    publication_lag_bdays=pub_lag,
                )
            finally:
                con.close()
        except Exception as exc:
            news_error = str(exc)
            logger.exception("GKG ingest failed: %s", exc)

    return SharedIngestResult(
        macro_ok=macro_ok,
        news_ok=news_ok,
        macro_error=macro_error,
        news_error=news_error,
    )


# ---------------------------------------------------------------------------
# Per-asset ingest (prices only)
# ---------------------------------------------------------------------------

def ingest_for_asset(
    asset_id: str,
    source_ticker: str,
    datasources_config: str,
    db_path: str | None = None,
    shared_ingest: SharedIngestResult | None = None,
) -> IngestResult:
    """Refresh price data for one asset and return its IngestResult.

    The macro and GKG availability flags are taken from *shared_ingest*
    (produced by run_shared_ingest).  If *shared_ingest* is None (e.g. during
    unit tests), macro_ok and news_ok default to True.

    Parameters
    ----------
    asset_id:
        Semantic asset identifier (for logging).
    source_ticker:
        Raw ticker symbol to refresh (e.g. "^GSPC").
    datasources_config:
        Path to config/prod/datasources.rpi5.yaml.
    db_path:
        Override path for financial_data.duckdb.
    shared_ingest:
        Result from run_shared_ingest; provides macro_ok / news_ok.
    """
    cfg = _load_datasources_config(datasources_config)
    resolved_db = _resolve_db_path(cfg, db_path)
    Path(resolved_db).parent.mkdir(parents=True, exist_ok=True)

    prices_ok = False
    prices_error: str | None = None
    data_cutoff_date = date.today() - timedelta(days=1)  # default to D-1

    try:
        price_cfg = PriceIngestConfig(
            db_path=resolved_db,
            default_start=str(cfg["prices"].get("default_start", "2010-01-01")),
            auto_adjust=bool(cfg["prices"].get("auto_adjust", True)),
            lookback_buffer_days=int(cfg["prices"].get("lookback_buffer_days", 5)),
        )
        res = refresh_prices(price_cfg, tickers=[source_ticker])
        if res.get("errors", {}).get(source_ticker):
            prices_error = res["errors"][source_ticker]
            logger.warning("[%s] Price ingest error: %s", asset_id, prices_error)

        # Determine freshness from DB (even if refresh errored, prior data may exist).
        # Only open read-only when the file already exists.
        if Path(resolved_db).exists():
            con = duckdb.connect(resolved_db, read_only=True)
            try:
                prices_ok, last_date = _check_prices_freshness(con, source_ticker)
                if last_date is not None:
                    # Cap at D-1: yfinance may return intraday/incomplete bars for today
                    # while the market is open, but features rely on closed daily data
                    # and macro publication lags assume D-1.
                    data_cutoff_date = min(last_date, date.today() - timedelta(days=1))
            finally:
                con.close()

    except Exception as exc:
        prices_error = str(exc)
        logger.exception("[%s] Price ingest failed: %s", asset_id, exc)

    # Derive macro / news availability from shared ingest result
    if shared_ingest is not None:
        macro_ok = shared_ingest.macro_ok
        news_ok = shared_ingest.news_ok
    else:
        # Fallback: trust DB freshness directly (no shared ingest provided)
        try:
            con = duckdb.connect(resolved_db, read_only=True)
            try:
                macro_ok = _check_macro_freshness(con)
                pub_lag = int(cfg.get("gkg", {}).get("publication_lag_bdays", 1))
                gkg_table = str(cfg.get("gkg", {}).get("daily_table", "news_features_daily"))
                news_ok = _check_gkg_freshness(con, table=gkg_table, publication_lag_bdays=pub_lag)
            finally:
                con.close()
        except Exception:
            macro_ok = False
            news_ok = False

    return IngestResult(
        prices_ok=prices_ok,
        macro_ok=macro_ok,
        news_ok=news_ok,
        data_cutoff_date=data_cutoff_date,
        prices_error=prices_error,
    )
