"""
Tests for src/quant_risk/prod/worker/features.py — per-asset production feature path.

Coverage:
- _build_features_cfg: YAML → BuildFeaturesConfig translation
- build_features_for_asset: happy path, missing ticker, missing macro,
  forecast_date not in calendar, feature_contract filtering, no cross-asset columns.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml

from quant_risk.prod.worker.features import FeaturesResult, _build_features_cfg, build_features_for_asset


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@dataclass
class _MockIngestResult:
    """Minimal stand-in for IngestResult — only data_cutoff_date is used."""
    data_cutoff_date: date
    prices_ok: bool = True
    macro_ok: bool = True
    news_ok: bool = True
    prices_error: Optional[str] = None


def _make_features_yaml(tmp_dir: Path, news_enabled: bool = False) -> str:
    """Write a minimal features.rpi5.yaml to *tmp_dir* and return its path."""
    cfg = {
        "calendar": {"freq": "B"},
        "targets": {"horizons": [5], "regime_bins": 3},
        "features": {
            "returns": {"type": "log", "lags": [1, 5, 20]},
            "realized_vol": {"window_sizes": [5, 20]},
            "macro_transforms": {"method": "diff", "lags": [1, 5, 20]},
            "macro_publication_lags": {"vix": 0, "fed_assets": 1},
            "shock_features": {
                "rv_long_window": 60,
                "rv_ema_spans": [10, 30],
                "vol_of_vol_window": 20,
                "return_shock_window": 60,
                "return_shock_quantiles": [0.8, 0.9],
                "volume_z_window": 20,
            },
        },
        "news_features": {
            "enabled": news_enabled,
            "source_table": "news_features_daily",
            "topics_enabled": [],
            "windows": [3, 10, 20],
            "attn_z_window": 252,
            "attn_shock_threshold": 2.0,
            "shock_windows": [3, 10, 20],
            "compact_mode": True,
            "topic_share_enabled": True,
            "include_roll_sum": False,
            "include_roll_mean": False,
            "include_roll_std": False,
            "include_tone_std": False,
            "include_tone_neg_share": False,
            "include_interactions": True,
            "health": {
                "min_nonzero_rate": 0.01,
                "min_unique_values": 3,
                "drop_dead_queries": True,
                "fail_if_all_dead": False,  # avoid RuntimeError in minimal DBs
            },
        },
    }
    path = tmp_dir / "features.rpi5.yaml"
    path.write_text(yaml.dump(cfg))
    return str(path)


def _populate_db(
    db_path: str,
    ticker: str = "^GSPC",
    n_days: int = 300,
    end_date: date | None = None,
    include_news: bool = False,
) -> date:
    """
    Seed a DuckDB file with raw_prices + macro_features (+ optionally news_features_daily).
    Returns the last price date inserted.
    """
    if end_date is None:
        end_date = date(2024, 6, 30)

    # Generate business days as Python date objects (avoid TIMESTAMP_NS → DATE issues)
    dates = [d.date() for d in pd.bdate_range(end=pd.Timestamp(end_date), periods=n_days)]

    con = duckdb.connect(db_path)

    # raw_prices
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_prices (
            ticker VARCHAR,
            date DATE,
            close DOUBLE,
            volume DOUBLE
        )
    """)
    prices_df = pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "close": 100.0 * np.cumprod(1 + np.random.normal(0, 0.01, n_days)),
        "volume": np.random.uniform(1e6, 1e7, n_days),
    })
    con.execute("INSERT INTO raw_prices SELECT * FROM prices_df")

    # macro_features
    con.execute("""
        CREATE TABLE IF NOT EXISTS macro_features (
            date DATE,
            vix DOUBLE,
            fed_assets DOUBLE
        )
    """)
    macro_df = pd.DataFrame({
        "date": dates,
        "vix": np.random.uniform(10, 40, n_days),
        "fed_assets": np.random.uniform(7e12, 9e12, n_days),
    })
    con.execute("INSERT INTO macro_features SELECT * FROM macro_df")

    if include_news:
        con.execute("""
            CREATE TABLE IF NOT EXISTS news_features_daily (
                query_id VARCHAR,
                date DATE,
                news_count DOUBLE,
                tone_mean DOUBLE,
                tone_std DOUBLE,
                tone_neg_share DOUBLE
            )
        """)
        news_df = pd.DataFrame({
            "query_id": "default",
            "date": dates,
            "news_count": np.random.randint(0, 50, n_days).astype(float),
            "tone_mean": np.random.uniform(-5, 5, n_days),
            "tone_std": np.random.uniform(0, 3, n_days),
            "tone_neg_share": np.random.uniform(0, 1, n_days),
        })
        con.execute("INSERT INTO news_features_daily SELECT * FROM news_df")

    con.close()
    return dates[-1]


# ---------------------------------------------------------------------------
# _build_features_cfg tests
# ---------------------------------------------------------------------------

class TestBuildFeaturesCfg:
    def test_rv_windows_mapped(self, tmp_path):
        path = _make_features_yaml(tmp_path)
        import yaml as _yaml
        cfg = _build_features_cfg(_yaml.safe_load(Path(path).read_text()), "/tmp/db")
        assert cfg.rv_windows == (5, 20)

    def test_macro_lags_mapped(self, tmp_path):
        path = _make_features_yaml(tmp_path)
        import yaml as _yaml
        cfg = _build_features_cfg(_yaml.safe_load(Path(path).read_text()), "/tmp/db")
        assert cfg.macro_lags == (1, 5, 20)

    def test_rv_long_window(self, tmp_path):
        path = _make_features_yaml(tmp_path)
        import yaml as _yaml
        cfg = _build_features_cfg(_yaml.safe_load(Path(path).read_text()), "/tmp/db")
        assert cfg.rv_long_window == 60

    def test_rv_ema_spans(self, tmp_path):
        path = _make_features_yaml(tmp_path)
        import yaml as _yaml
        cfg = _build_features_cfg(_yaml.safe_load(Path(path).read_text()), "/tmp/db")
        assert cfg.rv_ema_spans == (10, 30)

    def test_news_disabled_by_default_in_fixture(self, tmp_path):
        path = _make_features_yaml(tmp_path, news_enabled=False)
        import yaml as _yaml
        cfg = _build_features_cfg(_yaml.safe_load(Path(path).read_text()), "/tmp/db")
        assert cfg.news_enabled is False

    def test_news_enabled_flag(self, tmp_path):
        path = _make_features_yaml(tmp_path, news_enabled=True)
        import yaml as _yaml
        cfg = _build_features_cfg(_yaml.safe_load(Path(path).read_text()), "/tmp/db")
        assert cfg.news_enabled is True

    def test_compact_mode_true(self, tmp_path):
        path = _make_features_yaml(tmp_path)
        import yaml as _yaml
        cfg = _build_features_cfg(_yaml.safe_load(Path(path).read_text()), "/tmp/db")
        assert cfg.news_compact_mode is True

    def test_macro_publication_lags(self, tmp_path):
        path = _make_features_yaml(tmp_path)
        import yaml as _yaml
        cfg = _build_features_cfg(_yaml.safe_load(Path(path).read_text()), "/tmp/db")
        assert cfg.macro_publication_lags == {"vix": 0, "fed_assets": 1}

    def test_db_path_passed_through(self, tmp_path):
        path = _make_features_yaml(tmp_path)
        import yaml as _yaml
        cfg = _build_features_cfg(_yaml.safe_load(Path(path).read_text()), "/custom/path.duckdb")
        assert cfg.db_path == "/custom/path.duckdb"

    def test_no_cross_corr_window_impact(self, tmp_path):
        """cross_corr_window not set from YAML — default in BuildFeaturesConfig is fine."""
        path = _make_features_yaml(tmp_path)
        import yaml as _yaml
        cfg = _build_features_cfg(_yaml.safe_load(Path(path).read_text()), "/tmp/db")
        # The production function never calls the cross-asset block so the value doesn't matter.
        assert isinstance(cfg.cross_corr_window, int)


# ---------------------------------------------------------------------------
# build_features_for_asset — happy path
# ---------------------------------------------------------------------------

class TestBuildFeaturesForAssetHappyPath:
    def test_returns_features_result(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )
        assert isinstance(result, FeaturesResult)

    def test_feature_date_matches_cutoff(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )
        assert result.feature_date == last_date

    def test_single_row_returned(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )
        assert len(result.features) == 1

    def test_feature_count_positive(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )
        assert result.feature_count > 0
        assert result.feature_count == len(result.features.columns)

    def test_no_missing_columns_by_default(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )
        assert result.missing_columns == []

    def test_index_is_forecast_date(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )
        assert pd.Timestamp(last_date) in result.features.index

    def test_no_ticker_column_in_output(self, tmp_path):
        """Raw non-feature columns must be stripped."""
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )
        assert "ticker" not in result.features.columns
        assert "close" not in result.features.columns
        assert "volume" not in result.features.columns


# ---------------------------------------------------------------------------
# No cross-asset columns
# ---------------------------------------------------------------------------

class TestNoCrossAssetFeatures:
    def test_no_corr_columns(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )
        cols = result.features.columns.tolist()
        cross_cols = [c for c in cols if c.startswith("corr_")]
        assert cross_cols == [], f"Cross-asset corr columns found: {cross_cols}"

    def test_no_rv20_diff_columns(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )
        cols = result.features.columns.tolist()
        cross_cols = [c for c in cols if c.startswith("rv20_diff_") or c.startswith("rv20_ratio_")]
        assert cross_cols == [], f"Cross-asset rv20 columns found: {cross_cols}"

    def test_no_risk_off_proxy_columns(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )
        cols = result.features.columns.tolist()
        proxy_cols = [c for c in cols if "risk_off_proxy" in c or "risk_on_proxy" in c]
        assert proxy_cols == [], f"Cross-asset proxy columns found: {proxy_cols}"


# ---------------------------------------------------------------------------
# Per-asset feature columns present
# ---------------------------------------------------------------------------

class TestExpectedFeatureColumns:
    def _get_result(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        return build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )

    def test_logret_present(self, tmp_path):
        r = self._get_result(tmp_path)
        assert "logret" in r.features.columns

    def test_rv_windows_present(self, tmp_path):
        r = self._get_result(tmp_path)
        assert "rv_5" in r.features.columns
        assert "rv_20" in r.features.columns

    def test_logret_lags_present(self, tmp_path):
        r = self._get_result(tmp_path)
        assert "logret_lag1" in r.features.columns
        assert "logret_lag5" in r.features.columns
        assert "logret_lag20" in r.features.columns

    def test_rv_long_window_alias(self, tmp_path):
        r = self._get_result(tmp_path)
        assert "vol_60" in r.features.columns

    def test_vol_slope_present(self, tmp_path):
        r = self._get_result(tmp_path)
        assert "rv_slope_20_60" in r.features.columns
        assert "rv_ratio_20_60" in r.features.columns

    def test_volume_z_present(self, tmp_path):
        r = self._get_result(tmp_path)
        assert "volume_z_w20" in r.features.columns

    def test_macro_diff_present(self, tmp_path):
        r = self._get_result(tmp_path)
        # vix_diff should be present (macro transform + no pub lag → shift 0)
        assert "vix_diff" in r.features.columns


# ---------------------------------------------------------------------------
# feature_contract filtering
# ---------------------------------------------------------------------------

class TestFeatureContract:
    def test_contract_filters_columns(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        contract = ["logret", "rv_5", "rv_20"]
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
            feature_contract=contract,
        )
        assert list(result.features.columns) == contract

    def test_missing_contract_columns_reported(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        contract = ["logret", "rv_5", "nonexistent_feature_xyz"]
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
            feature_contract=contract,
        )
        assert "nonexistent_feature_xyz" in result.missing_columns

    def test_contract_none_no_filtering(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
            feature_contract=None,
        )
        # Should produce many columns
        assert result.feature_count > 5

    def test_empty_contract_treated_as_no_filter(self, tmp_path):
        """An empty list contract (falsy) is treated the same as None — no filtering."""
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
            feature_contract=[],
        )
        # [] is falsy → no filtering applied, all features returned
        assert result.feature_count > 0
        assert result.missing_columns == []


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestBuildFeaturesErrors:
    def test_missing_ticker_raises(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        _populate_db(db, ticker="^GSPC", n_days=300)
        ingest = _MockIngestResult(data_cutoff_date=date(2024, 6, 30))
        with pytest.raises(RuntimeError, match="No price data"):
            build_features_for_asset(
                asset_id="bitcoin",
                source_ticker="BTC-USD",  # not in DB
                ingest_result=ingest,
                features_config=features_cfg,
                db_path=db,
            )

    def test_forecast_date_beyond_data_raises(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path)
        _populate_db(db, ticker="^GSPC", n_days=300, end_date=date(2024, 6, 30))
        # Ask for a date well beyond available data
        future_date = date(2030, 1, 2)  # a Thursday
        ingest = _MockIngestResult(data_cutoff_date=future_date)
        with pytest.raises(RuntimeError):
            build_features_for_asset(
                asset_id="us_equities",
                source_ticker="^GSPC",
                ingest_result=ingest,
                features_config=features_cfg,
                db_path=db,
            )

    def test_db_not_exist_raises(self, tmp_path):
        features_cfg = _make_features_yaml(tmp_path)
        ingest = _MockIngestResult(data_cutoff_date=date(2024, 6, 28))
        with pytest.raises(Exception):
            build_features_for_asset(
                asset_id="us_equities",
                source_ticker="^GSPC",
                ingest_result=ingest,
                features_config=features_cfg,
                db_path=str(tmp_path / "nonexistent.duckdb"),
            )


# ---------------------------------------------------------------------------
# News features integration (news_enabled=True)
# ---------------------------------------------------------------------------

class TestNewsFeatures:
    def test_news_columns_present_when_enabled(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path, news_enabled=True)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300, include_news=True)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )
        # With news enabled, attn_z should be present
        assert "attn_z" in result.features.columns

    def test_news_interaction_columns_present(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path, news_enabled=True)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300, include_news=True)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )
        assert "attn_z_x_rv20" in result.features.columns

    def test_news_absent_when_disabled(self, tmp_path):
        db = str(tmp_path / "fin.duckdb")
        features_cfg = _make_features_yaml(tmp_path, news_enabled=False)
        last_date = _populate_db(db, ticker="^GSPC", n_days=300, include_news=True)
        ingest = _MockIngestResult(data_cutoff_date=last_date)
        result = build_features_for_asset(
            asset_id="us_equities",
            source_ticker="^GSPC",
            ingest_result=ingest,
            features_config=features_cfg,
            db_path=db,
        )
        news_cols = [c for c in result.features.columns if "attn_z" in c or "tone_" in c]
        assert news_cols == []
