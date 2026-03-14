'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-11

@description: Leakage tests for GKG daily features and as-of alignment.
'''

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from quant_risk.data.gkg import init_daily_schema, init_raw_schema, rebuild_daily_from_raw, upsert_gkg_raw
from quant_risk.features.build import BuildFeaturesConfig, build_features


def _build_test_prices(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA"] * len(dates),
            "date": dates,
            "close": np.linspace(100.0, 140.0, len(dates)),
            "volume": np.linspace(1000.0, 1500.0, len(dates)),
        }
    )


def _build_test_macro(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "vix": np.linspace(12.0, 25.0, len(dates))})


def test_gkg_features_do_not_change_past_when_future_news_changes(tmp_path):
    db_path = tmp_path / "gkg_no_leakage.duckdb"
    dates = pd.bdate_range("2024-01-01", periods=70)
    cutoff = pd.Timestamp("2024-03-01")

    con = duckdb.connect(str(db_path))
    try:
        prices_df = _build_test_prices(dates)
        macro_df = _build_test_macro(dates)
        con.register("tmp_prices", prices_df)
        con.register("tmp_macro", macro_df)
        con.execute("CREATE TABLE raw_prices AS SELECT * FROM tmp_prices")
        con.execute("CREATE TABLE macro_features AS SELECT * FROM tmp_macro")

        init_raw_schema(con, "gdelt_gkg_raw")
        init_daily_schema(con, "news_features_daily")

        base_raw = pd.DataFrame(
            {
                "gkg_id": [f"id_{i}" for i in range(len(dates))],
                "date": dates.date,
                "datetime": dates + pd.Timedelta(hours=12),
                "source": ["Reuters"] * len(dates),
                "url": [f"https://example.com/{i}" for i in range(len(dates))],
                "language": ["en"] * len(dates),
                "tone": np.linspace(-0.3, 0.3, len(dates)),
                "themes": ["macro"] * len(dates),
                "locations": [None] * len(dates),
                "persons": [None] * len(dates),
                "organizations": [None] * len(dates),
                "query_id": ["macro_us"] * len(dates),
                "source_file": ["mock"] * len(dates),
            }
        )
        upsert_gkg_raw(con, "gdelt_gkg_raw", base_raw)
        rebuild_daily_from_raw(
            con,
            raw_table="gdelt_gkg_raw",
            daily_table="news_features_daily",
            start_day=dates.min().date(),
            end_day=dates.max().date(),
            query_ids=["macro_us"],
        )
    finally:
        con.close()

    cfg = BuildFeaturesConfig(
        db_path=str(db_path),
        rv_windows=(2,),
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
        news_query_ids=("macro_us",),
        news_attn_z_window=10,
        news_shock_windows=(3,),
        news_health_min_unique_values=1,
        news_health_drop_dead_queries=False,
        news_health_fail_if_all_dead=False,
    )
    build_features(cfg, tickers=["AAA"])

    con = duckdb.connect(str(db_path))
    try:
        before = con.execute(
            """
            SELECT date, news_count_macro_us, news_count_mean_w3_macro_us
            FROM features_daily
            WHERE ticker='AAA' AND date <= ?
            ORDER BY date
            """,
            [cutoff],
        ).fetchdf()
    finally:
        con.close()

    con = duckdb.connect(str(db_path))
    try:
        future_mask = dates > cutoff
        perturb = pd.DataFrame(
            {
                "gkg_id": [f"id_future_{i}" for i, ok in enumerate(future_mask) if ok],
                "date": [d.date() for d, ok in zip(dates, future_mask) if ok],
                "datetime": [d + pd.Timedelta(hours=18) for d, ok in zip(dates, future_mask) if ok],
                "source": ["Reuters"] * int(future_mask.sum()),
                "url": [f"https://example.com/future/{i}" for i in range(int(future_mask.sum()))],
                "language": ["en"] * int(future_mask.sum()),
                "tone": [3.0] * int(future_mask.sum()),
                "themes": ["macro"] * int(future_mask.sum()),
                "locations": [None] * int(future_mask.sum()),
                "persons": [None] * int(future_mask.sum()),
                "organizations": [None] * int(future_mask.sum()),
                "query_id": ["macro_us"] * int(future_mask.sum()),
                "source_file": ["mock"] * int(future_mask.sum()),
            }
        )
        upsert_gkg_raw(con, "gdelt_gkg_raw", perturb)
        rebuild_daily_from_raw(
            con,
            raw_table="gdelt_gkg_raw",
            daily_table="news_features_daily",
            start_day=dates.min().date(),
            end_day=dates.max().date(),
            query_ids=["macro_us"],
        )
    finally:
        con.close()

    build_features(cfg, tickers=["AAA"])

    con = duckdb.connect(str(db_path))
    try:
        after = con.execute(
            """
            SELECT date, news_count_macro_us, news_count_mean_w3_macro_us
            FROM features_daily
            WHERE ticker='AAA' AND date <= ?
            ORDER BY date
            """,
            [cutoff],
        ).fetchdf()
    finally:
        con.close()

    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
