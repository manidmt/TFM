'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2025-12-22

@description: Module to fetch financial macroeconomic data from various sources.
'''

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import duckdb
import pandas as pd
from pandas_datareader import data as web  # FRED
import yfinance as yf


@dataclass(frozen=True)
class MacroIngestConfig:
    db_path: str
    table: str = "macro_features"
    default_start: str = "2010-01-01"
    lookback_buffer_days: int = 10


FRED_SERIES = {
    "VIXCLS": "vix",
    "WALCL": "fed_assets",
    "WTREGEN": "tga",
    "RRPONTSYD": "rrp",
    "M2SL": "m2",
    "DFF": "ffr",
    "SOFR": "sofr",
}


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def init_schema(con: duckdb.DuckDBPyConnection, table: str) -> None:
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            date DATE NOT NULL,
            vix DOUBLE,
            fed_assets DOUBLE,
            tga DOUBLE,
            rrp DOUBLE,
            m2 DOUBLE,
            ffr DOUBLE,
            sofr DOUBLE,
            net_liquidity DOUBLE,
            source TEXT,
            inserted_at TIMESTAMP,
            PRIMARY KEY (date)
        );
    """)


def get_last_date(con: duckdb.DuckDBPyConnection, table: str) -> Optional[date]:
    row = con.execute(f"SELECT MAX(date) FROM {table}").fetchone()
    return row[0] if row and row[0] is not None else None


def fetch_fred(start: str, end: Optional[str]) -> pd.DataFrame:
    df = web.DataReader(list(FRED_SERIES.keys()), "fred", start, end)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns=FRED_SERIES).reset_index().rename(columns={"DATE": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Example: net_liquidity = fed_assets - tga - rrp
    # Note: in FRED, WALCL usually in "millions of dollars" (depends on series),
    # and tga and rrp may be in different units.
    if {"fed_assets", "tga", "rrp"}.issubset(df.columns):
        df["net_liquidity"] = df["fed_assets"] - df["tga"] - df["rrp"]

    return df


def upsert_macro(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, source: str) -> int:
    if df.empty:
        return 0

    cols = [
        "date", "vix", "fed_assets", "tga", "rrp", "m2", "ffr", "sofr", "net_liquidity"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    df2 = df[cols].copy()
    df2["source"] = source
    df2["inserted_at"] = datetime.utcnow()

    con.register("tmp_macro", df2)

    con.execute(f"""
        INSERT INTO {table} (date, vix, fed_assets, tga, rrp, m2, ffr, sofr, net_liquidity, source, inserted_at)
        SELECT date, vix, fed_assets, tga, rrp, m2, ffr, sofr, net_liquidity, source, inserted_at
        FROM tmp_macro
        ON CONFLICT (date) DO UPDATE SET
            vix = excluded.vix,
            fed_assets = excluded.fed_assets,
            tga = excluded.tga,
            rrp = excluded.rrp,
            m2 = excluded.m2,
            ffr = excluded.ffr,
            sofr = excluded.sofr,
            net_liquidity = excluded.net_liquidity,
            source = excluded.source,
            inserted_at = excluded.inserted_at
    """)
    return len(df2)


def refresh_macro(cfg: MacroIngestConfig, end: Optional[str] = None) -> dict:
    con = connect(cfg.db_path)
    try:
        init_schema(con, cfg.table)

        last = get_last_date(con, cfg.table)
        if last:
            start = (last - timedelta(days=cfg.lookback_buffer_days)).isoformat()
        else:
            start = cfg.default_start

        df_fred = pd.DataFrame()
        errors = {}

        try:
            df_fred = fetch_fred(start=start, end=end)
        except Exception as e:
            errors["fred"] = str(e)

        inserted = upsert_macro(con, cfg.table, df_fred, source="fred")

        return {"inserted_rows": inserted, "errors": errors, "start": start, "end": end}
    finally:
        con.close()
