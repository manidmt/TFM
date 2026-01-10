'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-10

@description: Module to build features from prices and macro data.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import duckdb
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class BuildFeaturesConfig:
    db_path: str
    prices_table: str = "raw_prices"
    macro_table: str = "macro_features"
    out_table: str = "features_daily"
    calendar_freq: str = "B"  # Business day frequency
    start: str | None = None
    end: str | None = None
    rv_windows: tuple[int, ...] = (5, 20)
    return_lags: tuple[int, ...] = (1, 5, 20)
    macro_lags: tuple[int, ...] = (1, 5, 20)
    macro_transform: str = "diff"   # diff | pct_change | logdiff

def _macro_transform(s: pd.Series, method: str) -> pd.Series:
    if method == "diff":
        return s.diff()
    elif method == "pct_change":
        return s.pct_change()
    elif method == "logdiff":
        return np.log(s).diff()
    else:
        raise ValueError(f"Unknown macro transform method: {method}")
    
def build_features(cfg: BuildFeaturesConfig, tickers: Iterable[str]) -> dict:
    con = duckdb.connect(cfg.db_path)
    try:
        # Load prices
        dfp = con.execute(
            f"""
            SELECT ticker, date, close, volume
            FROM {cfg.prices_table}
            WHERE ticker IN ({",".join(["?"] * len(list(tickers)))})
            ORDER BY ticker, date
            """,
            list(tickers),
        ).df()
    
        if dfp.empty:
            raise RuntimeError("No price data loaded from raw_prices for requested tickers.")

        dfp["date"] = pd.to_datetime(dfp["date"])

        # Load macro
        dfm = con.execute(
            f"""SELECT * FROM {cfg.macro_table} ORDER BY date"""
        ).df()

        if dfm.empty:
            raise RuntimeError("No macro data loaded from macro_features.")
        
        dfm["date"] = pd.to_datetime(dfm["date"])

        min_by_ticker = dfp.groupby("ticker")["date"].min()
        max_by_ticker = dfp.groupby("ticker")["date"].max()

        start = max(min_by_ticker.max(), dfm["date"].min())
        end = min(max_by_ticker.min(), dfm["date"].max())

        if cfg.start:
            start = max(start, pd.to_datetime(cfg.start))
        if cfg.end:
            end = min(end, pd.to_datetime(cfg.end))

        master_idx = pd.date_range(start=start, end=end, freq=cfg.calendar_freq)

        # Macro transforms + lags
        dfm2 = dfm.set_index("date").reindex(master_idx).ffill()
        dfm2.index.name = "date"
        dfm2 = dfm2.reset_index()

        macro_cols = ["vix", "fed_assets", "tga", "rrp", "m2", "ffr", "sofr", "net_liquidity"]
        macro_cols = [c for c in macro_cols if c in dfm2.columns]
        for c in macro_cols:
            dfm2[f"{c}_{cfg.macro_transform}"] = _macro_transform(dfm2[c].astype(float), cfg.macro_transform)

        for c in macro_cols:
            tc = f"{c}_{cfg.macro_transform}"
            for lag in cfg.macro_lags:
                dfm2[f"{tc}_lag{lag}"] = dfm2[tc].shift(lag)

        # Build per ticker features aligned to master calendar
        all_out = []
        for t in tickers:
            sub = dfp[dfp["ticker"] == t].copy()
            sub = sub.set_index("date").reindex(master_idx).ffill()  # BTC gets projected to B-days
            sub.index.name = "date"
            sub = sub.reset_index()
            sub["ticker"] = t

            # log returns
            sub["logret"] = np.log(sub["close"]).diff()

            # lags of returns
            for lag in cfg.return_lags:
                sub[f"logret_lag{lag}"] = sub["logret"].shift(lag)

            # realized vol proxies (rolling std of logret)
            for w in cfg.rv_windows:
                sub[f"rv_{w}"] = sub["logret"].rolling(w).std()

            # Merge macro (already on master_idx)
            merged = sub.merge(dfm2, on="date", how="left")

            all_out.append(merged)

        out = pd.concat(all_out, ignore_index=True)

        # Drop rows with insufficient history (basic)
        min_lag = max(max(cfg.return_lags, default=0), max(cfg.macro_lags, default=0), max(cfg.rv_windows, default=0))
        out = out.sort_values(["ticker", "date"])
        out["rownum"] = out.groupby("ticker").cumcount()
        out = out[out["rownum"] >= min_lag].drop(columns=["rownum"])

        # Write to DuckDB
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {cfg.out_table} AS
            SELECT * FROM out LIMIT 0
        """)
        con.register("tmp_features", out)
        con.execute(f"DELETE FROM {cfg.out_table}")  # regenerate deterministically
        con.execute(f"INSERT INTO {cfg.out_table} SELECT * FROM tmp_features")

        return {
            "rows": len(out),
            "start": str(out["date"].min().date()),
            "end": str(out["date"].max().date()),
            "tickers": list(tickers),
            "out_table": cfg.out_table,
        }
    
    finally:
        con.close()