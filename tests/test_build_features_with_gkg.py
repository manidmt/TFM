'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-11

@description: Integration test for building features with canonical GKG daily table.
'''

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from quant_risk.features.build import BuildFeaturesConfig, build_features


def test_build_features_merges_gkg_daily_columns(tmp_path):
    db_path = tmp_path / "build_features_gkg.duckdb"
    dates = pd.bdate_range("2024-01-01", periods=90)

    prices = pd.DataFrame(
        {
            "ticker": ["AAA"] * len(dates),
            "date": dates,
            "close": np.linspace(100.0, 150.0, len(dates)),
            "volume": np.linspace(1000.0, 1800.0, len(dates)),
        }
    )
    macro = pd.DataFrame({"date": dates, "vix": np.linspace(10.0, 20.0, len(dates))})

    news = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "query_id": "macro_us",
                    "news_count": np.linspace(100.0, 200.0, len(dates)),
                    "tone_mean": np.linspace(-0.2, 0.2, len(dates)),
                    "tone_std": np.linspace(0.1, 0.3, len(dates)),
                    "tone_neg_share": np.linspace(0.7, 0.3, len(dates)),
                }
            ),
            pd.DataFrame(
                {
                    "date": dates,
                    "query_id": "rates_yields",
                    "news_count": np.linspace(60.0, 140.0, len(dates)),
                    "tone_mean": np.linspace(-0.1, 0.1, len(dates)),
                    "tone_std": np.linspace(0.05, 0.2, len(dates)),
                    "tone_neg_share": np.linspace(0.6, 0.4, len(dates)),
                }
            ),
        ],
        ignore_index=True,
    )

    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE raw_prices AS SELECT * FROM prices")
        con.execute("CREATE TABLE macro_features AS SELECT * FROM macro")
        con.execute("CREATE TABLE news_features_daily AS SELECT * FROM news")
    finally:
        con.close()

    cfg = BuildFeaturesConfig(
        db_path=str(db_path),
        rv_windows=(5, 20),
        return_lags=(1, 5),
        macro_lags=(1, 5),
        macro_transform="diff",
        news_enabled=True,
        news_source_table="news_features_daily",
        news_publication_lag_bdays=1,
        news_windows=(3, 10),
        news_include_roll_sum=True,
        news_include_roll_mean=True,
        news_include_roll_std=True,
        news_include_tone_std=True,
        news_include_tone_neg_share=True,
        news_attn_z_window=20,
        news_shock_windows=(3, 10),
        news_query_ids=("macro_us", "rates_yields"),
        news_include_interactions=True,
    )

    build_features(cfg, tickers=["AAA"])

    con = duckdb.connect(str(db_path))
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info('features_daily')").fetchall()]
        required = {
            "news_count_macro_us",
            "news_count_rates_yields",
            "log_news_count_macro_us",
            "attention_shock_z_macro_us",
            "tone_change_1d_macro_us",
            "tone_change_3d_macro_us",
            "neg_share_change_1d_macro_us",
            "attn_z_x_rv20_macro_us",
            "tone_change_1d_x_rv20_macro_us",
        }
        assert required.issubset(set(cols))

        check = con.execute(
            """
            SELECT
                news_count_macro_us,
                news_count_mean_w3_macro_us,
                tone_change_1d_macro_us,
                attn_z_x_rv20_macro_us
            FROM features_daily
            WHERE ticker='AAA'
            """
        ).fetchdf()
    finally:
        con.close()

    nan_ratio = check.isna().mean()
    assert float(nan_ratio["news_count_macro_us"]) < 0.20
    assert float(nan_ratio["news_count_mean_w3_macro_us"]) < 0.40
    assert float(nan_ratio["tone_change_1d_macro_us"]) < 0.25
    assert float(nan_ratio["attn_z_x_rv20_macro_us"]) < 0.60


def test_build_features_sanitizes_multiquery_suffixes_without_explicit_query_ids(tmp_path):
    db_path = tmp_path / "build_features_gkg_multiquery.duckdb"
    dates = pd.bdate_range("2024-01-01", periods=40)

    prices = pd.DataFrame(
        {
            "ticker": ["AAA"] * len(dates),
            "date": dates,
            "close": np.linspace(100.0, 120.0, len(dates)),
            "volume": np.linspace(1000.0, 1400.0, len(dates)),
        }
    )
    macro = pd.DataFrame({"date": dates, "vix": np.linspace(10.0, 15.0, len(dates))})

    news = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "query_id": "macro us",
                    "news_count": np.linspace(50.0, 90.0, len(dates)),
                    "tone_mean": np.linspace(-0.1, 0.1, len(dates)),
                    "tone_std": np.linspace(0.05, 0.15, len(dates)),
                    "tone_neg_share": np.linspace(0.6, 0.4, len(dates)),
                }
            ),
            pd.DataFrame(
                {
                    "date": dates,
                    "query_id": "rates-yields",
                    "news_count": np.linspace(30.0, 70.0, len(dates)),
                    "tone_mean": np.linspace(-0.05, 0.05, len(dates)),
                    "tone_std": np.linspace(0.03, 0.12, len(dates)),
                    "tone_neg_share": np.linspace(0.55, 0.45, len(dates)),
                }
            ),
        ],
        ignore_index=True,
    )

    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE raw_prices AS SELECT * FROM prices")
        con.execute("CREATE TABLE macro_features AS SELECT * FROM macro")
        con.execute("CREATE TABLE news_features_daily AS SELECT * FROM news")
    finally:
        con.close()

    cfg = BuildFeaturesConfig(
        db_path=str(db_path),
        rv_windows=(5,),
        return_lags=(1,),
        macro_lags=(1,),
        macro_transform="diff",
        news_enabled=True,
        news_source_table="news_features_daily",
        news_publication_lag_bdays=1,
        news_windows=(3,),
        news_include_roll_sum=False,
        news_include_roll_mean=True,
        news_include_roll_std=False,
    )

    build_features(cfg, tickers=["AAA"])

    con = duckdb.connect(str(db_path))
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info('features_daily')").fetchall()}
    finally:
        con.close()

    assert "news_count_macro_us" in cols
    assert "news_count_rates_yields" in cols
