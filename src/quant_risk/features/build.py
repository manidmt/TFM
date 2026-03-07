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
    macro_publication_lags: dict[str, int] | None = None
    rv_long_window: int = 60
    rv_ema_spans: tuple[int, int] = (10, 30)
    vol_of_vol_window: int = 20
    return_shock_window: int = 60
    return_shock_quantiles: tuple[float, ...] = (0.8, 0.9)
    volume_z_window: int = 20
    cross_corr_window: int = 20


def _rolling_percentile_last(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    last = values[-1]
    return float(np.mean(values <= last))

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
        tickers = list(tickers)
        # Load prices
        dfp = con.execute(
            f"""
            SELECT ticker, date, close, volume
            FROM {cfg.prices_table}
            WHERE ticker IN ({",".join(["?"] * len(tickers))})
            ORDER BY ticker, date
            """,
            tickers,
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
        publication_lags = cfg.macro_publication_lags or {
            "vix": 0,
            "fed_assets": 1,
            "tga": 1,
            "rrp": 1,
            "m2": 5,
            "ffr": 1,
            "sofr": 1,
            "net_liquidity": 1,
        }
        for c in macro_cols:
            lag_bdays = int(publication_lags.get(c, 0))
            if lag_bdays > 0:
                dfm2[c] = dfm2[c].shift(lag_bdays)
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

            # Realized-vol proxies (rolling std of logret).
            # Always compute the true 20-day series because downstream targets
            # and engineered features assume rv_20 semantics.
            rv_windows = {int(w) for w in cfg.rv_windows}
            rv_windows.update({20, int(cfg.rv_long_window)})
            for w in sorted(rv_windows):
                sub[f"rv_{w}"] = sub["logret"].rolling(w).std()

            # Alias for long-horizon realized vol so it is available as model feature
            # even if raw rv_* columns are filtered in dataset inference.
            sub[f"vol_{cfg.rv_long_window}"] = sub[f"rv_{cfg.rv_long_window}"]

            # Volatility acceleration / spread style features
            sub["rv_slope_20_60"] = sub["rv_20"] - sub[f"rv_{cfg.rv_long_window}"]
            sub["rv_ratio_20_60"] = sub["rv_20"] / sub[f"rv_{cfg.rv_long_window}"].replace(0.0, np.nan)
            sub["rv_logdiff_20_60"] = np.log(sub["rv_20"].clip(lower=1e-12)) - np.log(
                sub[f"rv_{cfg.rv_long_window}"].clip(lower=1e-12)
            )
            # Non-rv prefixed aliases for robust feature selection.
            sub["vol_slope_20_60"] = sub["rv_slope_20_60"]
            sub["vol_ratio_20_60"] = sub["rv_ratio_20_60"]
            sub["vol_logdiff_20_60"] = sub["rv_logdiff_20_60"]

            ema_fast, ema_slow = cfg.rv_ema_spans
            rv20_ema_fast = sub["rv_20"].ewm(span=ema_fast, adjust=False).mean()
            rv20_ema_slow = sub["rv_20"].ewm(span=ema_slow, adjust=False).mean()
            sub[f"rv20_ema_diff_{ema_fast}_{ema_slow}"] = rv20_ema_fast - rv20_ema_slow

            # Vol-of-vol
            sub["vol_of_vol_20"] = sub["rv_20"].rolling(cfg.vol_of_vol_window).std()

            # Return shock features
            sub["abs_logret"] = sub["logret"].abs()
            abs_ret = sub["abs_logret"]
            for q in cfg.return_shock_quantiles:
                q_name = int(round(float(q) * 100))
                q_col = f"abs_logret_q{q_name}_w{cfg.return_shock_window}"
                sub[q_col] = abs_ret.rolling(cfg.return_shock_window).quantile(float(q))
                sub[f"abs_logret_gt_q{q_name}_w{cfg.return_shock_window}"] = (
                    abs_ret > sub[q_col]
                ).astype(float)
            sub[f"abs_logret_pct_rank_w{cfg.return_shock_window}"] = abs_ret.rolling(
                cfg.return_shock_window
            ).apply(_rolling_percentile_last, raw=True)

            # Volume shock (z-score on log-volume)
            log_volume = np.log(sub["volume"].clip(lower=1.0))
            vol_mu = log_volume.rolling(cfg.volume_z_window).mean()
            vol_sd = log_volume.rolling(cfg.volume_z_window).std().replace(0.0, np.nan)
            sub[f"volume_z_w{cfg.volume_z_window}"] = (log_volume - vol_mu) / vol_sd
            sub[f"volume_z_abs_w{cfg.volume_z_window}"] = sub[f"volume_z_w{cfg.volume_z_window}"].abs()

            # Merge macro (already on master_idx)
            merged = sub.merge(dfm2, on="date", how="left")

            all_out.append(merged)

        out = pd.concat(all_out, ignore_index=True)

        # Cross-asset features on common calendar (no look-ahead, rolling past windows only)
        wide_logret = out.pivot(index="date", columns="ticker", values="logret").sort_index()
        wide_rv20 = out.pivot(index="date", columns="ticker", values="rv_20").sort_index()

        cross = pd.DataFrame(index=wide_logret.index)
        if "^GSPC" in wide_logret.columns and "BTC-USD" in wide_logret.columns:
            cross[f"corr_{cfg.cross_corr_window}_spx_btc"] = (
                wide_logret["^GSPC"]
                .rolling(cfg.cross_corr_window)
                .corr(wide_logret["BTC-USD"])
            )
        if "^GSPC" in wide_rv20.columns and "TLT" in wide_rv20.columns:
            cross["rv_spx_minus_tlt"] = wide_rv20["^GSPC"] - wide_rv20["TLT"]
            cross["risk_off_proxy_spx_tlt"] = (wide_rv20["^GSPC"] > wide_rv20["TLT"]).astype(float)
        if "BTC-USD" in wide_rv20.columns and "^GSPC" in wide_rv20.columns:
            cross["rv_btc_minus_spx"] = wide_rv20["BTC-USD"] - wide_rv20["^GSPC"]
            cross["risk_on_proxy_btc_spx"] = (wide_rv20["BTC-USD"] < wide_rv20["^GSPC"]).astype(float)

        cross = cross.reset_index()
        out = out.merge(cross, on="date", how="left")

        # Drop rows with insufficient history (basic)
        min_lag = max(max(cfg.return_lags, default=0), max(cfg.macro_lags, default=0), max(cfg.rv_windows, default=0))
        out = out.sort_values(["ticker", "date"])
        out["rownum"] = out.groupby("ticker").cumcount()
        out = out[out["rownum"] >= min_lag].drop(columns=["rownum"])

        # Write to DuckDB (recreate table to keep schema aligned with current feature set)
        con.register("tmp_features", out)
        con.execute(f"DROP TABLE IF EXISTS {cfg.out_table}")
        con.execute(f"CREATE TABLE {cfg.out_table} AS SELECT * FROM tmp_features")

        return {
            "rows": len(out),
            "start": str(out["date"].min().date()),
            "end": str(out["date"].max().date()),
            "tickers": list(tickers),
            "out_table": cfg.out_table,
        }
    
    finally:
        con.close()
