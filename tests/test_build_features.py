'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-10

@description: Tests for building features from prices and macro data.
'''

import duckdb
import pandas as pd
import numpy as np

from quant_risk.features.build import BuildFeaturesConfig, build_features

DB = "data/db/financial_data.duckdb"

def test_features_table_exists_and_has_rows():
    con = duckdb.connect(DB)
    try:
        n = con.execute("select count(*) from features_daily").fetchone()[0]
        assert n > 0
    finally:
        con.close()

def test_features_has_expected_tickers():
    con = duckdb.connect(DB)
    try:
        rows = con.execute("select distinct ticker from features_daily order by ticker").fetchall()
        tickers = [r[0] for r in rows]
        assert set(tickers) >= {"^GSPC", "BTC-USD", "TLT"}
    finally:
        con.close()

def test_features_calendar_is_business_day():
    con = duckdb.connect(DB)
    try:
        # weekend rows should be zero if you built on B-day calendar
        weekend = con.execute("""
            select count(*)
            from features_daily
            where strftime('%w', date) in ('0','6')
        """).fetchone()[0]
        assert weekend == 0
    finally:
        con.close()

def test_features_has_no_null_close_after_ffill():
    con = duckdb.connect(DB)
    try:
        n_null = con.execute("SELECT COUNT(*) FROM features_daily WHERE close IS NULL").fetchone()[0]
        assert n_null == 0, f"features_daily has {n_null} NULL close values."
    finally:
        con.close()


def test_features_has_required_columns():
    con = duckdb.connect(DB)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info('features_daily')").fetchall()]
        required = {"ticker", "date", "close", "logret", "rv_5", "rv_20"}
        missing = required - set(cols)
        assert not missing, f"features_daily missing columns: {missing}"
    finally:
        con.close()


def test_macro_publication_lag_is_applied(tmp_path):
    db_path = tmp_path / "lag_test.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        dates = pd.bdate_range("2021-01-01", periods=10)

        df_prices = pd.DataFrame(
            {
                "ticker": ["AAA"] * len(dates),
                "date": dates,
                "close": [100 + i for i in range(len(dates))],
                "volume": [1000] * len(dates),
            }
        )
        df_macro = pd.DataFrame(
            {
                "date": dates,
                "vix": [10 + i for i in range(len(dates))],
            }
        )
        con.execute("CREATE TABLE raw_prices AS SELECT * FROM df_prices")
        con.execute("CREATE TABLE macro_features AS SELECT * FROM df_macro")
    finally:
        con.close()

    cfg = BuildFeaturesConfig(
        db_path=str(db_path),
        rv_windows=(2,),
        return_lags=(1,),
        macro_lags=(1,),
        macro_transform="diff",
        macro_publication_lags={"vix": 2},
    )
    build_features(cfg, tickers=["AAA"])

    con = duckdb.connect(str(db_path))
    try:
        out = con.execute(
            """
            SELECT date, vix
            FROM features_daily
            WHERE ticker = 'AAA'
            ORDER BY date
            """
        ).df()
    finally:
        con.close()

    out["date"] = pd.to_datetime(out["date"])
    check_date = out["date"].iloc[0]
    expected_date = check_date - pd.offsets.BDay(2)
    expected_vix = float(df_macro.loc[df_macro["date"] == expected_date, "vix"].iloc[0])
    assert float(out["vix"].iloc[0]) == expected_vix


def test_rv20_is_true_window_even_if_not_requested(tmp_path):
    db_path = tmp_path / "rv20_true_window.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        dates = pd.bdate_range("2021-01-01", periods=45)
        close = [100.0 + 0.4 * i + (0.2 if i % 2 == 0 else -0.2) for i in range(len(dates))]
        df_prices = pd.DataFrame(
            {
                "ticker": ["AAA"] * len(dates),
                "date": dates,
                "close": close,
                "volume": [1000] * len(dates),
            }
        )
        df_macro = pd.DataFrame(
            {
                "date": dates,
                "vix": [20.0] * len(dates),
            }
        )
        con.execute("CREATE TABLE raw_prices AS SELECT * FROM df_prices")
        con.execute("CREATE TABLE macro_features AS SELECT * FROM df_macro")
    finally:
        con.close()

    cfg = BuildFeaturesConfig(
        db_path=str(db_path),
        rv_windows=(2,),
        return_lags=(1,),
        macro_lags=(1,),
    )
    build_features(cfg, tickers=["AAA"])

    con = duckdb.connect(str(db_path))
    try:
        out = con.execute(
            """
            SELECT date, rv_2, rv_20
            FROM features_daily
            WHERE ticker = 'AAA'
            ORDER BY date
            """
        ).df()
    finally:
        con.close()

    expected_rv20 = np.log(df_prices["close"]).diff().rolling(20).std().iloc[2:].reset_index(drop=True)
    pd.testing.assert_series_equal(out["rv_20"].reset_index(drop=True), expected_rv20, check_names=False)

    overlap = out["rv_2"].notna() & out["rv_20"].notna()
    assert overlap.any()
    assert (out.loc[overlap, "rv_20"] - out.loc[overlap, "rv_2"]).abs().gt(1e-12).any()
