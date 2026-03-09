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
