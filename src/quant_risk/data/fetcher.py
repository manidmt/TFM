'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2025-12-08

@description: Module to fetch financial data from various sources.
'''

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional

import duckdb
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class PriceIngestConfig:
    db_path: str
    table: str = "raw_prices"
    lookback_buffer_days: int = 5
    default_start: str = "2010-01-01"
    auto_adjust: bool = True
    timeout_seconds: int = 30


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def init_schema(con: duckdb.DuckDBPyConnection, table: str) -> None:
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            source TEXT,
            inserted_at TIMESTAMP,
            PRIMARY KEY (ticker, date)
        );
    """)


def get_last_dates(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, date]:
    """
    Returns last available date per ticker present in DB.
    """
    rows = con.execute(f"SELECT ticker, MAX(date) AS last_date FROM {table} GROUP BY ticker").fetchall()
    out: dict[str, date] = {}
    for t, d in rows:
        if d is not None:
            out[str(t)] = d
    return out


def download_ohlcv(
    tickers: Iterable[str],
    start: str,
    end: Optional[str],
    auto_adjust: bool,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    """
    Download OHLCV for tickers using yfinance.
    Returns a DataFrame with columns: ticker, date, open, high, low, close, volume
    """
    df = yf.download(
        list(tickers),
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
        timeout=timeout_seconds,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])

    records = []
    if isinstance(df.columns, pd.MultiIndex):
        for t in df.columns.levels[0]:
            if t not in df.columns.get_level_values(0):
                continue
            sub = df[t].copy()
            if sub.empty:
                continue
            sub = sub.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
            })
            sub["ticker"] = t
            sub = sub.reset_index().rename(columns={"Date": "date"})
            records.append(sub[["ticker", "date", "open", "high", "low", "close", "volume"]])
        out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    else:
        sub = df.copy().rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
        })
        sub = sub.reset_index().rename(columns={"Date": "date"})
        out = sub.assign(ticker=list(tickers)[0])[["ticker", "date", "open", "high", "low", "close", "volume"]]

    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.dropna(subset=["date"]).sort_values(["ticker", "date"])


def upsert_prices(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, source: str = "yfinance") -> int:
    if df.empty:
        return 0

    df2 = df.copy()
    df2["source"] = source
    df2["inserted_at"] = datetime.now(timezone.utc)

    con.register("tmp_prices", df2)

    con.execute(f"""
        INSERT INTO {table} (ticker, date, open, high, low, close, volume, source, inserted_at)
        SELECT ticker, date, open, high, low, close, volume, source, inserted_at
        FROM tmp_prices
        ON CONFLICT (ticker, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low  = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            source = excluded.source,
            inserted_at = excluded.inserted_at
    """)
    return len(df2)


def refresh_prices(cfg: PriceIngestConfig, tickers: list[str], end: Optional[str] = None) -> dict:
    con = connect(cfg.db_path)
    try:
        init_schema(con, cfg.table)

        last = get_last_dates(con, cfg.table)
        today = date.today()

        inserted = 0
        errors: dict[str, str] = {}

        for t in tickers:
            last_date = last.get(t)
            if last_date:
                start_date = (last_date - timedelta(days=cfg.lookback_buffer_days)).isoformat()
            else:
                start_date = cfg.default_start

            try:
                df = download_ohlcv(
                    [t],
                    start=start_date,
                    end=end,
                    auto_adjust=cfg.auto_adjust,
                    timeout_seconds=cfg.timeout_seconds,
                )
                if df.empty:
                    errors[t] = (
                        "Empty download result from yfinance "
                        f"(start={start_date}, end={end or 'today'})."
                    )
                    continue
                inserted += upsert_prices(con, cfg.table, df, source="yfinance")
            except Exception as e:
                errors[t] = str(e)

        return {"inserted_rows": inserted, "errors": errors, "as_of": today.isoformat()}
    finally:
        con.close()
