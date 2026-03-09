'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-08

@description: Smoke tests for GDELT ingestion.
'''

from __future__ import annotations

import duckdb
import pandas as pd

from quant_risk.data.gdelt import (
    GdeltIngestConfig,
    refresh_gdelt,
    upsert_gdelt_daily,
)


def test_refresh_gdelt_smoke_with_mocked_fetch(tmp_path, monkeypatch):
    db_path = tmp_path / "gdelt_ingest.duckdb"

    def _fake_fetch(_cfg: GdeltIngestConfig, day, query=None):
        return pd.DataFrame(
            {
                "date": [day, day, day],
                "tone": [-1.0, 0.0, 1.0],
            }
        )

    monkeypatch.setattr("quant_risk.data.gdelt.fetch_day_articles", _fake_fetch)

    cfg = GdeltIngestConfig(
        db_path=str(db_path),
        start="2024-01-01",
        lookback_buffer_days=2,
        mode="artlist",
    )
    res = refresh_gdelt(cfg, end="2024-01-03")
    assert res["inserted_rows"] == 3
    assert not res["errors"]

    con = duckdb.connect(str(db_path))
    try:
        n_rows = con.execute("SELECT COUNT(*) FROM gdelt_gkg_daily").fetchone()[0]
        assert n_rows == 3
        row = con.execute(
            """
            SELECT news_count, tone_mean, tone_neg_share
            FROM gdelt_gkg_daily
            WHERE date = DATE '2024-01-02'
            """
        ).fetchone()
        assert int(row[0]) == 3
        assert float(row[1]) == 0.0
        assert abs(float(row[2]) - (1.0 / 3.0)) < 1e-9
    finally:
        con.close()


def test_refresh_gdelt_timeline_mode_with_mocked_range_fetch(tmp_path, monkeypatch):
    db_path = tmp_path / "gdelt_ingest_timeline.duckdb"

    def _fake_vol(_cfg: GdeltIngestConfig, start_day, end_day, query=None):
        return pd.DataFrame(
            {
                "date": [start_day, end_day],
                "news_count": [100.0, 120.0],
            }
        )

    def _fake_tone(_cfg: GdeltIngestConfig, start_day, end_day, query=None):
        return pd.DataFrame(
            {
                "date": [start_day, end_day],
                "tone_mean": [0.25, -0.10],
            }
        )

    monkeypatch.setattr("quant_risk.data.gdelt.fetch_timeline_volraw", _fake_vol)
    monkeypatch.setattr("quant_risk.data.gdelt.fetch_timeline_tone", _fake_tone)

    cfg = GdeltIngestConfig(
        db_path=str(db_path),
        start="2024-01-01",
        lookback_buffer_days=2,
        mode="timeline",
    )
    res = refresh_gdelt(cfg, end="2024-01-03")
    assert res["inserted_rows"] == 2
    assert not res["errors"]
    assert res["mode"] == "timeline"

    con = duckdb.connect(str(db_path))
    try:
        n_rows = con.execute("SELECT COUNT(*) FROM gdelt_gkg_daily").fetchone()[0]
        assert n_rows == 2
        row = con.execute(
            """
            SELECT query_id, news_count, tone_mean, tone_std, tone_neg_share
            FROM gdelt_gkg_daily
            WHERE date = DATE '2024-01-03'
            """
        ).fetchone()
        assert row[0] == "default"
        assert float(row[1]) == 120.0
        assert float(row[2]) == -0.10
        assert row[3] is None
        assert row[4] is None
    finally:
        con.close()


def test_refresh_gdelt_multiquery_writes_query_id_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "gdelt_ingest_multiquery.duckdb"

    def _fake_vol(_cfg: GdeltIngestConfig, start_day, end_day, query=None):
        base = 10.0 if "bitcoin" in str(query).lower() else 100.0
        return pd.DataFrame({"date": [start_day, end_day], "news_count": [base, base + 1.0]})

    def _fake_tone(_cfg: GdeltIngestConfig, start_day, end_day, query=None):
        base = -0.2 if "bitcoin" in str(query).lower() else 0.2
        return pd.DataFrame({"date": [start_day, end_day], "tone_mean": [base, base + 0.05]})

    monkeypatch.setattr("quant_risk.data.gdelt.fetch_timeline_volraw", _fake_vol)
    monkeypatch.setattr("quant_risk.data.gdelt.fetch_timeline_tone", _fake_tone)

    cfg = GdeltIngestConfig(
        db_path=str(db_path),
        start="2024-01-01",
        lookback_buffer_days=2,
        mode="timeline",
        queries={
            "macro_us": "(stocks OR treasury)",
            "crypto": "(bitcoin OR crypto)",
        },
    )
    res = refresh_gdelt(cfg, end="2024-01-03")
    assert res["inserted_rows"] == 4
    assert res["query_count"] == 2
    assert sorted(res["query_ids"]) == ["crypto", "macro_us"]

    con = duckdb.connect(str(db_path))
    try:
        n_rows = con.execute("SELECT COUNT(*) FROM gdelt_gkg_daily").fetchone()[0]
        assert n_rows == 4
        qids = [r[0] for r in con.execute("SELECT DISTINCT query_id FROM gdelt_gkg_daily ORDER BY query_id").fetchall()]
        assert qids == ["crypto", "macro_us"]
    finally:
        con.close()


def test_upsert_gdelt_daily_updates_existing_date(tmp_path):
    db_path = tmp_path / "gdelt_upsert.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE gdelt_gkg_daily (
                date DATE NOT NULL,
                query_id TEXT NOT NULL,
                news_count BIGINT,
                tone_mean DOUBLE,
                tone_std DOUBLE,
                tone_neg_share DOUBLE,
                source TEXT,
                inserted_at TIMESTAMP,
                PRIMARY KEY (date, query_id)
            )
            """
        )
        first = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-05").date()],
                "query_id": ["default"],
                "news_count": [10],
                "tone_mean": [0.1],
                "tone_std": [0.2],
                "tone_neg_share": [0.4],
            }
        )
        second = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-05").date()],
                "query_id": ["default"],
                "news_count": [11],
                "tone_mean": [0.7],
                "tone_std": [0.5],
                "tone_neg_share": [0.2],
            }
        )
        upsert_gdelt_daily(con, "gdelt_gkg_daily", first)
        upsert_gdelt_daily(con, "gdelt_gkg_daily", second)

        row = con.execute(
            """
            SELECT query_id, news_count, tone_mean, tone_std, tone_neg_share
            FROM gdelt_gkg_daily
            WHERE date = DATE '2024-01-05' AND query_id='default'
            """
        ).fetchone()
        assert row[0] == "default"
        assert int(row[1]) == 11
        assert float(row[2]) == 0.7
        assert float(row[3]) == 0.5
        assert float(row[4]) == 0.2
    finally:
        con.close()
