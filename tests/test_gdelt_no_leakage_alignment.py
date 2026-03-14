'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-08

@description: Tests for GDELT feature alignment without look-ahead leakage.
'''

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from quant_risk.features.build import BuildFeaturesConfig, build_features


def test_gdelt_features_apply_publication_lag_before_rolling(tmp_path):
    db_path = tmp_path / "gdelt_alignment.duckdb"
    dates = pd.bdate_range("2024-01-01", periods=40)

    df_prices = pd.DataFrame(
        {
            "ticker": ["AAA"] * len(dates),
            "date": dates,
            "close": np.linspace(100.0, 140.0, len(dates)),
            "volume": np.linspace(1000.0, 1200.0, len(dates)),
        }
    )
    df_macro = pd.DataFrame({"date": dates, "vix": np.linspace(10.0, 20.0, len(dates))})
    df_gdelt = pd.DataFrame(
        {
            "date": dates,
            "news_count": np.arange(len(dates), dtype=float),
            "tone_mean": np.linspace(-0.2, 0.3, len(dates)),
            "tone_std": np.linspace(0.1, 0.4, len(dates)),
            "tone_neg_share": np.linspace(0.8, 0.2, len(dates)),
        }
    )

    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE raw_prices AS SELECT * FROM df_prices")
        con.execute("CREATE TABLE macro_features AS SELECT * FROM df_macro")
        con.execute("CREATE TABLE gdelt_gkg_daily AS SELECT * FROM df_gdelt")
    finally:
        con.close()

    cfg = BuildFeaturesConfig(
        db_path=str(db_path),
        rv_windows=(2,),
        return_lags=(1,),
        macro_lags=(1,),
        macro_transform="diff",
        news_enabled=True,
        news_publication_lag_bdays=1,
        news_windows=(3,),
        news_include_roll_sum=False,
        news_include_roll_mean=True,
        news_include_roll_std=False,
        news_attn_z_window=5,
        news_shock_windows=(3,),
    )
    build_features(cfg, tickers=["AAA"])

    con = duckdb.connect(str(db_path))
    try:
        out = con.execute(
            """
            SELECT date, news_count, news_count_mean_w3
            FROM features_daily
            WHERE ticker = 'AAA'
            ORDER BY date
            """
        ).df()
    finally:
        con.close()

    out["date"] = pd.to_datetime(out["date"])
    expected = (
        df_gdelt.set_index("date")["news_count"]
        .shift(1)
        .rolling(3)
        .mean()
        .rename("news_count_mean_w3_expected")
    )
    lagged = df_gdelt.set_index("date")["news_count"].shift(1).rename("news_count_expected")
    current = df_gdelt.set_index("date")["news_count"].rename("news_count_current")

    check = out.merge(expected, left_on="date", right_index=True, how="left")
    check = check.merge(lagged, left_on="date", right_index=True, how="left")
    check = check.merge(current, left_on="date", right_index=True, how="left")

    pd.testing.assert_series_equal(
        check["news_count"].reset_index(drop=True),
        check["news_count_expected"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        check["news_count_mean_w3"].reset_index(drop=True),
        check["news_count_mean_w3_expected"].reset_index(drop=True),
        check_names=False,
    )
    assert not check["news_count"].equals(check["news_count_current"])


def test_gdelt_multiquery_pivots_per_query_id(tmp_path):
    db_path = tmp_path / "gdelt_alignment_multiquery.duckdb"
    dates = pd.bdate_range("2024-01-01", periods=25)

    df_prices = pd.DataFrame(
        {
            "ticker": ["AAA"] * len(dates),
            "date": dates,
            "close": np.linspace(100.0, 125.0, len(dates)),
            "volume": np.linspace(1000.0, 1100.0, len(dates)),
        }
    )
    df_macro = pd.DataFrame({"date": dates, "vix": np.linspace(10.0, 15.0, len(dates))})
    df_gdelt = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "query_id": "macro_us",
                    "news_count": np.arange(len(dates), dtype=float),
                    "tone_mean": np.linspace(0.1, 0.3, len(dates)),
                    "tone_std": 0.0,
                    "tone_neg_share": 0.0,
                }
            ),
            pd.DataFrame(
                {
                    "date": dates,
                    "query_id": "crypto",
                    "news_count": np.arange(len(dates), dtype=float) + 100.0,
                    "tone_mean": np.linspace(-0.4, -0.2, len(dates)),
                    "tone_std": 0.0,
                    "tone_neg_share": 0.0,
                }
            ),
        ],
        ignore_index=True,
    )

    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE raw_prices AS SELECT * FROM df_prices")
        con.execute("CREATE TABLE macro_features AS SELECT * FROM df_macro")
        con.execute("CREATE TABLE gdelt_gkg_daily AS SELECT * FROM df_gdelt")
    finally:
        con.close()

    cfg = BuildFeaturesConfig(
        db_path=str(db_path),
        rv_windows=(2,),
        return_lags=(1,),
        macro_lags=(1,),
        macro_transform="diff",
        news_enabled=True,
        news_publication_lag_bdays=1,
        news_windows=(3,),
        news_include_roll_sum=False,
        news_include_roll_mean=True,
        news_include_roll_std=False,
        news_query_ids=("macro_us", "crypto"),
        news_attn_z_window=5,
        news_shock_windows=(3,),
    )
    build_features(cfg, tickers=["AAA"])

    con = duckdb.connect(str(db_path))
    try:
        out = con.execute(
            """
            SELECT
                date,
                news_count_macro_us,
                news_count_crypto,
                news_count_mean_w3_macro_us,
                news_count_mean_w3_crypto
            FROM features_daily
            WHERE ticker = 'AAA'
            ORDER BY date
            """
        ).df()
    finally:
        con.close()

    out["date"] = pd.to_datetime(out["date"])
    expected_macro = (
        df_gdelt[df_gdelt["query_id"] == "macro_us"]
        .set_index("date")["news_count"]
        .shift(1)
    )
    expected_crypto = (
        df_gdelt[df_gdelt["query_id"] == "crypto"]
        .set_index("date")["news_count"]
        .shift(1)
    )
    merged = out.merge(expected_macro.rename("exp_macro"), left_on="date", right_index=True, how="left")
    merged = merged.merge(expected_crypto.rename("exp_crypto"), left_on="date", right_index=True, how="left")

    pd.testing.assert_series_equal(
        merged["news_count_macro_us"].reset_index(drop=True),
        merged["exp_macro"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        merged["news_count_crypto"].reset_index(drop=True),
        merged["exp_crypto"].reset_index(drop=True),
        check_names=False,
    )


def test_gdelt_tone_feature_toggles_exclude_and_include_columns(tmp_path):
    db_path = tmp_path / "gdelt_tone_toggles.duckdb"
    dates = pd.bdate_range("2024-01-01", periods=20)

    df_prices = pd.DataFrame(
        {
            "ticker": ["AAA"] * len(dates),
            "date": dates,
            "close": np.linspace(100.0, 120.0, len(dates)),
            "volume": np.linspace(1000.0, 1050.0, len(dates)),
        }
    )
    df_macro = pd.DataFrame({"date": dates, "vix": np.linspace(10.0, 15.0, len(dates))})
    df_gdelt = pd.DataFrame(
        {
            "date": dates,
            "query_id": "default",
            "news_count": np.arange(len(dates), dtype=float),
            "tone_mean": np.linspace(-0.1, 0.2, len(dates)),
            "tone_std": np.linspace(0.05, 0.15, len(dates)),
            "tone_neg_share": np.linspace(0.7, 0.3, len(dates)),
        }
    )

    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE raw_prices AS SELECT * FROM df_prices")
        con.execute("CREATE TABLE macro_features AS SELECT * FROM df_macro")
        con.execute("CREATE TABLE gdelt_gkg_daily AS SELECT * FROM df_gdelt")
    finally:
        con.close()

    cfg_off = BuildFeaturesConfig(
        db_path=str(db_path),
        rv_windows=(2,),
        return_lags=(1,),
        macro_lags=(1,),
        macro_transform="diff",
        news_enabled=True,
        news_windows=(3,),
        news_include_roll_sum=False,
        news_include_roll_mean=True,
        news_include_roll_std=False,
        news_include_tone_std=False,
        news_include_tone_neg_share=False,
        news_attn_z_window=5,
        news_shock_windows=(3,),
    )
    build_features(cfg_off, tickers=["AAA"])

    con = duckdb.connect(str(db_path))
    try:
        cols_off = [r[1] for r in con.execute("PRAGMA table_info('features_daily')").fetchall()]
    finally:
        con.close()

    assert "tone_std" not in cols_off
    assert "tone_neg_share" not in cols_off
    assert "tone_std_mean_w3" not in cols_off
    assert "tone_neg_share_mean_w3" not in cols_off

    cfg_on = BuildFeaturesConfig(
        db_path=str(db_path),
        rv_windows=(2,),
        return_lags=(1,),
        macro_lags=(1,),
        macro_transform="diff",
        news_enabled=True,
        news_windows=(3,),
        news_include_roll_sum=False,
        news_include_roll_mean=True,
        news_include_roll_std=False,
        news_include_tone_std=True,
        news_include_tone_neg_share=True,
        news_attn_z_window=5,
        news_shock_windows=(3,),
    )
    build_features(cfg_on, tickers=["AAA"])

    con = duckdb.connect(str(db_path))
    try:
        cols_on = [r[1] for r in con.execute("PRAGMA table_info('features_daily')").fetchall()]
    finally:
        con.close()

    assert "tone_std" in cols_on
    assert "tone_neg_share" in cols_on
    assert "tone_std_mean_w3" in cols_on
    assert "tone_neg_share_mean_w3" in cols_on


def test_gdelt_derived_attention_and_interaction_features(tmp_path):
    db_path = tmp_path / "gdelt_derived_features.duckdb"
    dates = pd.bdate_range("2024-01-01", periods=45)

    df_prices = pd.DataFrame(
        {
            "ticker": ["AAA"] * len(dates),
            "date": dates,
            "close": np.linspace(100.0, 130.0, len(dates)),
            "volume": np.linspace(1000.0, 1200.0, len(dates)),
        }
    )
    df_macro = pd.DataFrame({"date": dates, "vix": np.linspace(10.0, 18.0, len(dates))})
    df_gdelt = pd.DataFrame(
        {
            "date": dates,
            "query_id": "default",
            "news_count": np.arange(len(dates), dtype=float),
            "tone_mean": np.linspace(-0.3, 0.3, len(dates)),
            "tone_std": np.linspace(0.05, 0.20, len(dates)),
            "tone_neg_share": np.linspace(0.8, 0.2, len(dates)),
        }
    )

    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE raw_prices AS SELECT * FROM df_prices")
        con.execute("CREATE TABLE macro_features AS SELECT * FROM df_macro")
        con.execute("CREATE TABLE gdelt_gkg_daily AS SELECT * FROM df_gdelt")
    finally:
        con.close()

    cfg = BuildFeaturesConfig(
        db_path=str(db_path),
        rv_windows=(2,),
        return_lags=(1,),
        macro_lags=(1,),
        macro_transform="diff",
        news_enabled=True,
        news_publication_lag_bdays=1,
        news_windows=(3,),
        news_include_roll_sum=False,
        news_include_roll_mean=False,
        news_include_roll_std=False,
        news_attn_z_window=5,
        news_shock_windows=(3,),
        news_include_interactions=True,
    )
    build_features(cfg, tickers=["AAA"])

    con = duckdb.connect(str(db_path))
    try:
        out = con.execute(
            """
            SELECT
                date,
                news_count,
                log_news,
                attn_z,
                attn_shock,
                shock_days_w3,
                tone_change_1d,
                tone_mean_w3,
                tone_std_w3,
                attn_z_x_rv20,
                tone_change_1d_x_attn_z,
                abs_tone_mean_x_attn_z
            FROM features_daily
            WHERE ticker = 'AAA'
            ORDER BY date
            """
        ).df()
    finally:
        con.close()

    out["date"] = pd.to_datetime(out["date"])
    expected_log_news = np.log1p(df_gdelt.set_index("date")["news_count"]).shift(1).rename("expected_log_news")
    merged = out.merge(expected_log_news, left_on="date", right_index=True, how="left")

    pd.testing.assert_series_equal(
        merged["log_news"].reset_index(drop=True),
        merged["expected_log_news"].reset_index(drop=True),
        check_names=False,
    )
    assert out["attn_z_x_rv20"].notna().any()
    assert out["tone_change_1d_x_attn_z"].notna().any()
    assert out["abs_tone_mean_x_attn_z"].notna().any()
