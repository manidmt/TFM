'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-11

@description: Smoke tests for canonical GKG raw ingestion.
'''

from __future__ import annotations

import duckdb
import pandas as pd

from quant_risk.data.gkg import GkgIngestConfig, refresh_gkg


def test_refresh_gkg_smoke_with_mocked_raw_fetch(tmp_path, monkeypatch):
    db_path = tmp_path / "gkg_ingest.duckdb"

    def _fake_fetch(_cfg: GkgIngestConfig, *, start_day, end_day):
        df = pd.DataFrame(
            {
                "gkg_id": ["id1", "id2", "id3"],
                "date": [start_day, start_day, end_day],
                "datetime": [
                    pd.Timestamp(f"{start_day} 12:00:00"),
                    pd.Timestamp(f"{start_day} 12:30:00"),
                    pd.Timestamp(f"{end_day} 13:00:00"),
                ],
                "source": ["Reuters", "Bloomberg", "Reuters"],
                "url": [
                    "https://example.com/fed-inflation",
                    "https://example.com/bitcoin-rally",
                    "https://example.com/treasury-yields",
                ],
                "language": ["en", "en", "en"],
                "tone": [-1.0, 0.5, -0.2],
                "themes": ["FED;INFLATION", "CRYPTO;BITCOIN", "TREASURY;YIELD"],
                "locations": [None, None, None],
                "persons": [None, None, None],
                "organizations": [None, None, None],
                "source_file": ["mock://1", "mock://1", "mock://2"],
            }
        )
        return df, {}

    monkeypatch.setattr("quant_risk.data.gkg.fetch_gkg_raw_records", _fake_fetch)

    cfg = GkgIngestConfig(
        db_path=str(db_path),
        start="2024-01-01",
        store_raw=True,
        topics={
            "fed_inflation": "(Federal Reserve OR inflation OR fed)",
            "crypto_market": "(bitcoin OR crypto)",
            "rates_yields": "(treasury OR yield)",
        },
    )
    res = refresh_gkg(cfg, end="2024-01-03")

    assert res["inserted_raw_rows"] == 3
    assert res["inserted_daily_rows"] == 3
    assert not res["errors"]
    assert res["profile_name"] == "gkg_v1_light"
    assert res["profile_version"] == "2026-03-13"
    assert res["profile_mode"] == "light_daily"
    assert bool(res["topics_hash"])

    con = duckdb.connect(str(db_path))
    try:
        raw_cols = [r[1] for r in con.execute("PRAGMA table_info('gdelt_gkg_raw')").fetchall()]
        assert {"gkg_id", "date", "datetime", "source", "url", "language", "tone", "themes", "query_id"}.issubset(
            set(raw_cols)
        )

        n_raw = con.execute("SELECT COUNT(*) FROM gdelt_gkg_raw").fetchone()[0]
        n_daily = con.execute("SELECT COUNT(*) FROM news_features_daily").fetchone()[0]
        assert n_raw == 3
        assert n_daily == 3

        topic_ids = [r[0] for r in con.execute("SELECT DISTINCT query_id FROM gdelt_gkg_raw ORDER BY query_id").fetchall()]
        assert topic_ids == ["crypto_market", "fed_inflation", "rates_yields"]

        manifest = con.execute(
            """
            SELECT profile_name, profile_version, profile_mode, start_date, end_date, topic_ids, store_raw,
                   sample_interval_minutes, max_files_per_day, max_files_total, inserted_raw_rows,
                   inserted_daily_rows, error_count
            FROM gkg_ingest_runs
            """
        ).fetchdf()
        assert len(manifest) == 1
        row = manifest.iloc[0]
        assert row["profile_name"] == "gkg_v1_light"
        assert row["profile_version"] == "2026-03-13"
        assert row["profile_mode"] == "light_daily"
        assert row["topic_ids"] == "crypto_market|fed_inflation|rates_yields"
        assert bool(row["store_raw"]) is True
        assert int(row["inserted_raw_rows"]) == 3
        assert int(row["inserted_daily_rows"]) == 3
        assert int(row["error_count"]) == 0
    finally:
        con.close()


def test_refresh_gkg_light_mode_without_raw_storage(tmp_path, monkeypatch):
    db_path = tmp_path / "gkg_ingest_light.duckdb"

    def _fake_fetch(_cfg: GkgIngestConfig, *, start_day, end_day):
        df = pd.DataFrame(
            {
                "gkg_id": ["id1", "id2"],
                "date": [start_day, end_day],
                "datetime": [pd.Timestamp(f"{start_day} 12:00:00"), pd.Timestamp(f"{end_day} 12:00:00")],
                "source": ["Reuters", "Reuters"],
                "url": ["https://example.com/inflation", "https://example.com/bitcoin"],
                "language": ["en", "en"],
                "tone": [-0.2, 0.4],
                "themes": ["INFLATION", "CRYPTO"],
                "locations": [None, None],
                "persons": [None, None],
                "organizations": [None, None],
                "source_file": ["mock://1", "mock://2"],
            }
        )
        return df, {}

    monkeypatch.setattr("quant_risk.data.gkg.fetch_gkg_raw_records", _fake_fetch)

    cfg = GkgIngestConfig(
        db_path=str(db_path),
        start="2024-01-01",
        store_raw=False,
        topics={
            "macro_us": "(inflation)",
            "crypto_market": "(bitcoin)",
        },
    )
    res = refresh_gkg(cfg, end="2024-01-02")
    assert res["inserted_raw_rows"] == 0
    assert res["inserted_daily_rows"] == 2
    assert res["store_raw"] is False
    assert res["profile_name"] == "gkg_v1_light"

    con = duckdb.connect(str(db_path))
    try:
        # raw table should not exist in light mode
        n_raw_tbl = con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema='main' AND table_name='gdelt_gkg_raw'
            """
        ).fetchone()[0]
        assert int(n_raw_tbl) == 0

        n_daily = con.execute("SELECT COUNT(*) FROM news_features_daily").fetchone()[0]
        assert int(n_daily) == 2

        manifest = con.execute(
            """
            SELECT profile_name, store_raw, inserted_raw_rows, inserted_daily_rows, error_count
            FROM gkg_ingest_runs
            """
        ).fetchdf()
        assert len(manifest) == 1
        row = manifest.iloc[0]
        assert row["profile_name"] == "gkg_v1_light"
        assert bool(row["store_raw"]) is False
        assert int(row["inserted_raw_rows"]) == 0
        assert int(row["inserted_daily_rows"]) == 2
        assert int(row["error_count"]) == 0
    finally:
        con.close()
