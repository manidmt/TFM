'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-11

@description: Tests for daily feature aggregation from canonical GKG raw table.
'''

from __future__ import annotations

import duckdb
import pandas as pd

from quant_risk.data.gkg import init_daily_schema, init_raw_schema, rebuild_daily_from_raw, upsert_gkg_raw


def test_rebuild_daily_from_raw_preserves_multiple_query_ids(tmp_path):
    db_path = tmp_path / "gkg_daily.duckdb"

    con = duckdb.connect(str(db_path))
    try:
        init_raw_schema(con, "gdelt_gkg_raw")
        init_daily_schema(con, "news_features_daily")

        raw = pd.DataFrame(
            {
                "gkg_id": ["a1", "a2", "b1", "b2"],
                "date": [
                    pd.Timestamp("2024-01-02").date(),
                    pd.Timestamp("2024-01-02").date(),
                    pd.Timestamp("2024-01-02").date(),
                    pd.Timestamp("2024-01-03").date(),
                ],
                "datetime": [
                    pd.Timestamp("2024-01-02 10:00:00"),
                    pd.Timestamp("2024-01-02 12:00:00"),
                    pd.Timestamp("2024-01-02 09:00:00"),
                    pd.Timestamp("2024-01-03 11:00:00"),
                ],
                "source": ["Reuters", "Reuters", "CoinDesk", "CoinDesk"],
                "url": ["u1", "u2", "u3", "u4"],
                "language": ["en", "en", "en", "en"],
                "tone": [-0.5, 0.5, 0.2, -0.2],
                "themes": ["FED", "CPI", "CRYPTO", "CRYPTO"],
                "locations": [None, None, None, None],
                "persons": [None, None, None, None],
                "organizations": [None, None, None, None],
                "query_id": ["macro_us", "macro_us", "crypto_market", "crypto_market"],
                "source_file": ["mock", "mock", "mock", "mock"],
            }
        )
        upsert_gkg_raw(con, "gdelt_gkg_raw", raw)

        n_daily = rebuild_daily_from_raw(
            con,
            raw_table="gdelt_gkg_raw",
            daily_table="news_features_daily",
            start_day=pd.Timestamp("2024-01-01").date(),
            end_day=pd.Timestamp("2024-01-05").date(),
        )
        assert n_daily == 3

        daily_cols = [r[1] for r in con.execute("PRAGMA table_info('news_features_daily')").fetchall()]
        assert {"date", "query_id", "news_count", "tone_mean", "tone_std", "tone_neg_share"}.issubset(
            set(daily_cols)
        )

        rows = con.execute(
            """
            SELECT date, query_id, news_count, tone_mean, tone_neg_share
            FROM news_features_daily
            ORDER BY date, query_id
            """
        ).fetchdf()

        assert rows["query_id"].tolist() == ["crypto_market", "macro_us", "crypto_market"]
        macro_row = rows[rows["query_id"] == "macro_us"].iloc[0]
        assert int(macro_row["news_count"]) == 2
        assert abs(float(macro_row["tone_mean"]) - 0.0) < 1e-12
        assert abs(float(macro_row["tone_neg_share"]) - 0.5) < 1e-12
    finally:
        con.close()
