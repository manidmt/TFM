'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-10

@description: Module to build features from prices and macro data.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import itertools
import re

import duckdb
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class BuildFeaturesConfig:
    db_path: str
    prices_table: str = "raw_prices"
    macro_table: str = "macro_features"
    news_source_table: str = "news_features_daily"
    # Deprecated fallback for legacy DOC timeline experiments.
    gdelt_table: str = "gdelt_gkg_daily"
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
    news_enabled: bool = False
    news_publication_lag_bdays: int = 1
    news_windows: tuple[int, ...] = (3, 10, 20)
    news_include_roll_sum: bool = True
    news_include_roll_mean: bool = True
    news_include_roll_std: bool = True
    news_include_tone_std: bool = False
    news_include_tone_neg_share: bool = False
    news_attn_z_window: int = 252
    news_attn_shock_threshold: float = 2.0
    news_shock_windows: tuple[int, ...] = (3, 10, 20)
    news_compact_mode: bool = False
    news_topic_share_enabled: bool = True
    news_health_min_nonzero_rate: float = 0.01
    news_health_min_unique_values: int = 3
    news_health_drop_dead_queries: bool = True
    news_health_fail_if_all_dead: bool = True
    news_include_interactions: bool = True
    news_query_ids: tuple[str, ...] = ()


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    """
    Check if a table exists in DuckDB main schema.
    """
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema='main' AND table_name=?
        """,
        [table],
    ).fetchone()
    return bool(row and int(row[0]) > 0)


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


def _sanitize_query_id(query_id: str) -> str:
    """
    Convert query_id to a safe lowercase suffix for feature names.
    """
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(query_id).strip().lower()).strip("_")
    return s or "default"


def _sanitize_ticker_id(ticker: str) -> str:
    """
    Convert ticker symbols into safe lowercase feature-name tokens.
    """
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(ticker).strip().lower()).strip("_")
    return s or "asset"


def _load_news_block(
    con: duckdb.DuckDBPyConnection,
    cfg: BuildFeaturesConfig,
    master_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build a date-aligned news feature block applying publication lag before rolling windows.
    """
    if not cfg.news_enabled:
        return pd.DataFrame({"date": master_idx})

    primary_table = str(cfg.news_source_table).strip()
    fallback_table = str(cfg.gdelt_table).strip()
    candidates: list[str] = []
    for t in (primary_table, fallback_table):
        if t and t not in candidates:
            candidates.append(t)

    selected_table = next((t for t in candidates if _table_exists(con, t)), None)
    if selected_table is None:
        raise RuntimeError(
            "No configured news table found while news_features.enabled=true. "
            f"Tried={candidates!r}"
        )
    if selected_table != primary_table:
        print(
            "[build_features][news] using legacy fallback table "
            f"{selected_table!r}; preferred source is {primary_table!r}."
        )

    try:
        table_cols = [r[1] for r in con.execute(f"PRAGMA table_info('{selected_table}')").fetchall()]
        has_query_id = "query_id" in table_cols
        if has_query_id:
            dfn = con.execute(
                f"""
                SELECT query_id, date, news_count, tone_mean, tone_std, tone_neg_share
                FROM {selected_table}
                ORDER BY query_id, date
                """
            ).df()
        else:
            dfn = con.execute(
                f"""
                SELECT 'default' AS query_id, date, news_count, tone_mean, tone_std, tone_neg_share
                FROM {selected_table}
                ORDER BY date
                """
            ).df()
    except Exception as exc:
        raise RuntimeError(
            f"News table '{selected_table}' is unavailable while news_features.enabled=true."
        ) from exc

    out = pd.DataFrame({"date": master_idx})

    configured_ids_raw = [str(q).strip() for q in cfg.news_query_ids if str(q).strip()]
    configured_ids: list[str] = []
    for query_id in configured_ids_raw:
        sanitized = _sanitize_query_id(query_id)
        if sanitized and sanitized not in configured_ids:
            configured_ids.append(sanitized)
    if dfn.empty:
        query_ids = configured_ids or ["default"]
        dfn = pd.DataFrame({"query_id": query_ids})
    else:
        dfn["query_id"] = dfn["query_id"].astype(str).map(_sanitize_query_id)
        dfn["date"] = pd.to_datetime(dfn["date"])
        dfn = (
            dfn.groupby(["query_id", "date"], as_index=False)
            .agg(
                news_count=("news_count", "sum"),
                tone_mean=("tone_mean", "mean"),
                tone_std=("tone_std", "mean"),
                tone_neg_share=("tone_neg_share", "mean"),
            )
            .sort_values(["query_id", "date"])
        )
        query_ids = configured_ids or sorted(dfn["query_id"].unique().tolist())
    use_suffix = len(query_ids) > 1 or len(configured_ids) > 0

    windows = tuple(sorted({int(w) for w in cfg.news_windows if int(w) > 0}))
    shock_windows = tuple(sorted({int(w) for w in cfg.news_shock_windows if int(w) > 0}))
    attn_z_window = max(2, int(cfg.news_attn_z_window))
    attn_shock_threshold = float(cfg.news_attn_shock_threshold)
    lag_bdays = int(cfg.news_publication_lag_bdays)
    internal_raw_cols = ["news_count", "tone_mean", "tone_std", "tone_neg_share"]
    raw_cols = ["news_count", "tone_mean"]
    if bool(cfg.news_include_tone_std):
        raw_cols.append("tone_std")
    if bool(cfg.news_include_tone_neg_share):
        raw_cols.append("tone_neg_share")
    base_cols = [
        *raw_cols,
        "log_news",
        "log_news_count",
        "attn_z",
        "attention_z",
        "attention_shock_z",
        "attn_shock",
        "tone_change_1d",
        "tone_change_3d",
        "neg_share_change_1d",
    ]
    if bool(cfg.news_topic_share_enabled):
        base_cols.extend(["topic_share", "topic_share_z"])

    aligned_sub_by_query: dict[str, pd.DataFrame] = {}
    health_rows: list[dict[str, float | int | str | bool]] = []

    for query_id in query_ids:
        if dfn.empty or "date" not in dfn.columns:
            sub = pd.DataFrame({"date": master_idx})
            for c in internal_raw_cols:
                sub[c] = 0.0
        else:
            sub = dfn[dfn["query_id"] == query_id].copy()
            sub = (
                sub.set_index("date")
                .reindex(master_idx)
                .reset_index()
                .rename(columns={"index": "date"})
            )
            sub["news_count"] = pd.to_numeric(sub["news_count"], errors="coerce").fillna(0.0)
            sub["tone_mean"] = pd.to_numeric(sub["tone_mean"], errors="coerce").fillna(0.0)
            sub["tone_std"] = pd.to_numeric(sub["tone_std"], errors="coerce").fillna(0.0)
            sub["tone_neg_share"] = pd.to_numeric(sub["tone_neg_share"], errors="coerce").fillna(0.0)

        if lag_bdays > 0:
            # Anti-leakage: lag is applied before any rolling/derived statistics.
            for c in internal_raw_cols:
                sub[c] = sub[c].shift(lag_bdays)
        aligned_sub_by_query[query_id] = sub

    active_query_ids: list[str] = []
    for query_id in query_ids:
        sub = aligned_sub_by_query[query_id]
        news_s = pd.to_numeric(sub["news_count"], errors="coerce")
        nonzero_rate = float((news_s.fillna(0.0) > 0.0).mean())
        n_unique = int(news_s.nunique(dropna=True))
        std = float(news_s.std(skipna=True) if news_s.notna().any() else 0.0)
        is_dead = bool(
            nonzero_rate < float(cfg.news_health_min_nonzero_rate)
            or n_unique < int(cfg.news_health_min_unique_values)
        )
        health_rows.append(
            {
                "query_id": str(query_id),
                "nonzero_rate": nonzero_rate,
                "n_unique_news_count": n_unique,
                "std_news_count": std,
                "is_dead": is_dead,
            }
        )
        if is_dead and bool(cfg.news_health_drop_dead_queries):
            print(
                "[build_features][news] dropping dead query_id="
                f"{query_id!r} nonzero_rate={nonzero_rate:.4f} n_unique={n_unique} std={std:.4f}"
            )
            continue
        active_query_ids.append(query_id)

    news_health = {
        "query_health": health_rows,
        "active_query_ids": [str(q) for q in active_query_ids],
        "dropped_query_ids": [
            r["query_id"]
            for r in health_rows
            if bool(r.get("is_dead", False)) and bool(cfg.news_health_drop_dead_queries)
        ],
    }
    out.attrs["news_health"] = news_health

    if not active_query_ids:
        if bool(cfg.news_health_fail_if_all_dead):
            raise RuntimeError(
                "All configured news query channels appear dead after alignment/lag checks; "
                "check query syntax or ingest coverage."
            )
        out.attrs["news_health"] = news_health
        return out

    total_news_count = np.zeros(len(master_idx), dtype=float)
    if bool(cfg.news_topic_share_enabled):
        for query_id in active_query_ids:
            s = pd.to_numeric(aligned_sub_by_query[query_id]["news_count"], errors="coerce").fillna(0.0)
            total_news_count += s.clip(lower=0.0).to_numpy(dtype=float)

    for query_id in active_query_ids:
        sub = aligned_sub_by_query[query_id].copy()

        sub["log_news"] = np.log1p(sub["news_count"].clip(lower=0.0))
        sub["log_news_count"] = sub["log_news"]
        roll_mu = sub["log_news"].rolling(attn_z_window).mean()
        roll_sd = sub["log_news"].rolling(attn_z_window).std().replace(0.0, np.nan)
        sub["attn_z"] = (sub["log_news"] - roll_mu) / roll_sd
        sub["attn_z"] = sub["attn_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sub["attention_z"] = sub["attn_z"]
        # Backward-compatible alias.
        sub["attention_shock_z"] = sub["attn_z"]
        sub["attn_shock"] = (sub["attn_z"] > attn_shock_threshold).astype(float)
        sub["tone_change_1d"] = sub["tone_mean"].diff(1)
        sub["tone_change_3d"] = sub["tone_mean"].diff(3)
        sub["neg_share_change_1d"] = sub["tone_neg_share"].diff(1)
        if bool(cfg.news_topic_share_enabled):
            numer = pd.to_numeric(sub["news_count"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
            den = np.where(total_news_count > 0.0, total_news_count, np.nan)
            share = np.divide(numer, den)
            share = np.nan_to_num(share, nan=0.0, posinf=0.0, neginf=0.0)
            sub["topic_share"] = share
            share_mu = sub["topic_share"].rolling(attn_z_window).mean()
            share_sd = sub["topic_share"].rolling(attn_z_window).std().replace(0.0, np.nan)
            sub["topic_share_z"] = (sub["topic_share"] - share_mu) / share_sd
            sub["topic_share_z"] = sub["topic_share_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        shock_cols: list[str] = []
        for w in shock_windows:
            c = f"shock_days_w{w}"
            sub[c] = sub["attn_shock"].rolling(w).sum()
            shock_cols.append(c)

        tone_window_cols: list[str] = []
        for w in windows:
            c_mu = f"tone_mean_w{w}"
            c_sd = f"tone_std_w{w}"
            sub[c_mu] = sub["tone_mean"].rolling(w).mean()
            sub[c_sd] = sub["tone_mean"].rolling(w).std()
            tone_window_cols.extend([c_mu, c_sd])

        if not bool(cfg.news_compact_mode):
            for c in base_cols:
                for w in windows:
                    roll = sub[c].rolling(w)
                    if cfg.news_include_roll_sum:
                        sub[f"{c}_sum_w{w}"] = roll.sum()
                    if cfg.news_include_roll_mean:
                        sub[f"{c}_mean_w{w}"] = roll.mean()
                    if cfg.news_include_roll_std:
                        sub[f"{c}_std_w{w}"] = roll.std()

        cols_to_merge = [c for c in [*base_cols, *shock_cols, *tone_window_cols] if c in sub.columns]
        if not bool(cfg.news_compact_mode):
            for c in base_cols:
                for w in windows:
                    if cfg.news_include_roll_sum:
                        name = f"{c}_sum_w{w}"
                        if name in sub.columns:
                            cols_to_merge.append(name)
                    if cfg.news_include_roll_mean:
                        name = f"{c}_mean_w{w}"
                        if name in sub.columns:
                            cols_to_merge.append(name)
                    if cfg.news_include_roll_std:
                        name = f"{c}_std_w{w}"
                        if name in sub.columns:
                            cols_to_merge.append(name)
        cols_to_merge = list(dict.fromkeys(cols_to_merge))
        sub = sub[["date", *cols_to_merge]].copy()
        if use_suffix:
            suffix = _sanitize_query_id(query_id)
            rename_map = {c: f"{c}_{suffix}" for c in cols_to_merge}
            sub = sub.rename(columns=rename_map)
        out = out.merge(sub, on="date", how="left")

    out.attrs["news_health"] = news_health
    return out
    
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
        news_block = _load_news_block(con, cfg, master_idx)
        news_health = dict(news_block.attrs.get("news_health", {}))
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
            if cfg.news_enabled:
                merged = merged.merge(news_block, on="date", how="left")
                if bool(cfg.news_include_interactions):
                    attn_cols = [c for c in merged.columns if c == "attn_z" or c.startswith("attn_z_")]
                    sigma_col = None
                    for sigma_candidate in ("garch_sigma_t", "sigma_t"):
                        if sigma_candidate in merged.columns:
                            sigma_col = sigma_candidate
                            break
                    for attn_col in attn_cols:
                        suffix = "" if attn_col == "attn_z" else attn_col[len("attn_z") :]
                        merged[f"attn_z_x_rv20{suffix}"] = merged[attn_col] * merged["rv_20"]
                        if "rv_5" in merged.columns:
                            merged[f"attn_z_x_rv5{suffix}"] = merged[attn_col] * merged["rv_5"]
                        if sigma_col is not None:
                            merged[f"attn_z_x_garch_sigma_t{suffix}"] = merged[attn_col] * merged[sigma_col]

                        tone_chg_col = f"tone_change_1d{suffix}"
                        if tone_chg_col in merged.columns:
                            merged[f"tone_change_1d_x_attn_z{suffix}"] = merged[tone_chg_col] * merged[attn_col]
                            merged[f"tone_change_1d_x_rv20{suffix}"] = merged[tone_chg_col] * merged["rv_20"]

                        tone_mean_col = f"tone_mean{suffix}"
                        if tone_mean_col in merged.columns:
                            merged[f"abs_tone_mean_x_attn_z{suffix}"] = merged[tone_mean_col].abs() * merged[attn_col]
                        tone_neg_col = f"tone_neg_share{suffix}"
                        if sigma_col is not None and tone_neg_col in merged.columns:
                            merged[f"tone_neg_share_x_garch_sigma_t{suffix}"] = (
                                merged[tone_neg_col] * merged[sigma_col]
                            )
                    topic_cols = [
                        c for c in merged.columns if c == "topic_share_z" or c.startswith("topic_share_z_")
                    ]
                    for topic_col in topic_cols:
                        suffix = "" if topic_col == "topic_share_z" else topic_col[len("topic_share_z") :]
                        merged[f"topic_share_z_x_rv20{suffix}"] = merged[topic_col] * merged["rv_20"]
                        if "rv_5" in merged.columns:
                            merged[f"topic_share_z_x_rv5{suffix}"] = merged[topic_col] * merged["rv_5"]
                        tone_chg_col = f"tone_change_1d{suffix}"
                        if tone_chg_col in merged.columns:
                            merged[f"tone_change_1d_x_topic_share_z{suffix}"] = (
                                merged[tone_chg_col] * merged[topic_col]
                            )

            all_out.append(merged)

        out = pd.concat(all_out, ignore_index=True)

        # Cross-asset features on common calendar (no look-ahead, rolling past windows only)
        wide_logret = out.pivot(index="date", columns="ticker", values="logret").sort_index()
        wide_rv20 = out.pivot(index="date", columns="ticker", values="rv_20").sort_index()

        cross = pd.DataFrame(index=wide_logret.index)
        available_tickers = sorted(
            set(wide_logret.columns.astype(str)).intersection(wide_rv20.columns.astype(str)),
            key=_sanitize_ticker_id,
        )
        for left, right in itertools.combinations(available_tickers, 2):
            left_id = _sanitize_ticker_id(left)
            right_id = _sanitize_ticker_id(right)
            pair_suffix = f"{left_id}_{right_id}"

            cross[f"corr_{cfg.cross_corr_window}_{pair_suffix}"] = (
                wide_logret[left].rolling(cfg.cross_corr_window).corr(wide_logret[right])
            )
            cross[f"rv20_diff_{pair_suffix}"] = wide_rv20[left] - wide_rv20[right]
            denom = wide_rv20[right].replace(0.0, np.nan)
            cross[f"rv20_ratio_{pair_suffix}"] = wide_rv20[left] / denom
            cross[f"rv20_gt_{pair_suffix}"] = (wide_rv20[left] > wide_rv20[right]).astype(float)

        # Backward-compatible aliases for the original hand-crafted pairs.
        for corr_alias in ("corr_20_gspc_btc_usd", "corr_20_btc_usd_gspc"):
            if corr_alias in cross.columns and cfg.cross_corr_window == 20:
                cross["corr_20_spx_btc"] = cross[corr_alias]
                break
        for diff_alias, gt_alias in (
            ("rv20_diff_gspc_tlt", "rv20_gt_gspc_tlt"),
            ("rv20_diff_tlt_gspc", "rv20_gt_tlt_gspc"),
        ):
            if diff_alias in cross.columns:
                if diff_alias == "rv20_diff_tlt_gspc":
                    cross["rv_spx_minus_tlt"] = -cross[diff_alias]
                else:
                    cross["rv_spx_minus_tlt"] = cross[diff_alias]
                cross["risk_off_proxy_spx_tlt"] = np.where(
                    cross["rv_spx_minus_tlt"].isna(),
                    np.nan,
                    (cross["rv_spx_minus_tlt"] > 0.0).astype(float),
                )
                break
        for diff_alias, gt_alias in (
            ("rv20_diff_btc_usd_gspc", "rv20_gt_btc_usd_gspc"),
            ("rv20_diff_gspc_btc_usd", "rv20_gt_gspc_btc_usd"),
        ):
            if diff_alias in cross.columns:
                if diff_alias == "rv20_diff_gspc_btc_usd":
                    cross["rv_btc_minus_spx"] = -cross[diff_alias]
                else:
                    cross["rv_btc_minus_spx"] = cross[diff_alias]
                cross["risk_on_proxy_btc_spx"] = np.where(
                    cross["rv_btc_minus_spx"].isna(),
                    np.nan,
                    (cross["rv_btc_minus_spx"] < 0.0).astype(float),
                )
                break

        cross = cross.reset_index()
        out = out.merge(cross, on="date", how="left")

        # Drop rows with insufficient history (basic)
        news_lag = 0
        if cfg.news_enabled:
            news_hist = [int(w) for w in cfg.news_windows if int(w) > 0]
            news_hist += [int(w) for w in cfg.news_shock_windows if int(w) > 0]
            news_hist += [int(cfg.news_attn_z_window)]
            news_lag = int(cfg.news_publication_lag_bdays) + max(news_hist or [0])
        min_lag = max(
            max(cfg.return_lags, default=0),
            max(cfg.macro_lags, default=0),
            max(cfg.rv_windows, default=0),
            news_lag,
        )
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
            "news_health": news_health,
        }
    
    finally:
        con.close()
