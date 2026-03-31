'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Production feature extraction adapter — per-asset, no cross-asset.

Extracts the per-asset production feature vector from financial_data.duckdb using
the locked production feature profile (gkg_v1_features_compact).
No cross-asset dependencies are allowed here (rpi5.md §7).

Design:
  - Adapts src/quant_risk/features/build.py for single-ticker inference.
  - The cross-asset section (lines 574–633 in build.py) is deliberately excluded.
  - Feature date is derived from IngestResult.data_cutoff_date (D-1 price bar).
  - If a feature_contract list is supplied, the output is filtered to those columns
    and missing_columns is populated for any gaps.

Reference: rpi5.md §7, §10.3, §26.4, §26.5
'''

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pandas as pd

from quant_risk.config import load_yaml
from quant_risk.features.build import (
    BuildFeaturesConfig,
    _load_news_block,
    _macro_transform,
    _rolling_percentile_last,
)

if TYPE_CHECKING:
    from quant_risk.prod.worker.ingest import IngestResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeaturesResult:
    """Outcome of a production feature extraction run for a single asset.

    Attributes
    ----------
    features:
        Single-row DataFrame (index = forecast_date) containing all features
        required by the active bundle's feature_contract.json.
    feature_count:
        Number of feature columns produced.
    feature_date:
        The date for which features were computed (equals ingest_result.data_cutoff_date).
    missing_columns:
        Any columns listed in the bundle's feature_contract that could not be
        populated.  Non-empty means the inference may be degraded.
    """

    features: pd.DataFrame
    feature_count: int
    feature_date: date
    missing_columns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_features_cfg(yaml_cfg: dict, db_path: str) -> BuildFeaturesConfig:
    """Translate features.rpi5.yaml into a BuildFeaturesConfig for production."""
    feat = yaml_cfg.get("features", {})
    shock = feat.get("shock_features", {})
    news = yaml_cfg.get("news_features", {})
    health = news.get("health", {})
    cal = yaml_cfg.get("calendar", {})

    pub_lags_raw = feat.get("macro_publication_lags", {})
    macro_publication_lags: dict[str, int] | None = (
        {str(k): int(v) for k, v in pub_lags_raw.items()} if pub_lags_raw else None
    )

    return BuildFeaturesConfig(
        db_path=db_path,
        prices_table="raw_prices",
        macro_table="macro_features",
        news_source_table=str(news.get("source_table", "news_features_daily")),
        calendar_freq=str(cal.get("freq", "B")),
        rv_windows=tuple(int(w) for w in feat.get("realized_vol", {}).get("window_sizes", [5, 20])),
        return_lags=tuple(int(l) for l in feat.get("returns", {}).get("lags", [1, 5, 20])),
        macro_lags=tuple(int(l) for l in feat.get("macro_transforms", {}).get("lags", [1, 5, 20])),
        macro_transform=str(feat.get("macro_transforms", {}).get("method", "diff")),
        macro_publication_lags=macro_publication_lags,
        rv_long_window=int(shock.get("rv_long_window", 60)),
        rv_ema_spans=tuple(int(s) for s in shock.get("rv_ema_spans", [10, 30])),
        vol_of_vol_window=int(shock.get("vol_of_vol_window", 20)),
        return_shock_window=int(shock.get("return_shock_window", 60)),
        return_shock_quantiles=tuple(float(q) for q in shock.get("return_shock_quantiles", [0.8, 0.9])),
        volume_z_window=int(shock.get("volume_z_window", 20)),
        # cross_corr_window intentionally left at default — never used in per-asset path
        news_enabled=bool(news.get("enabled", True)),
        news_publication_lag_bdays=int(news.get("publication_lag_bdays", 1)),
        news_windows=tuple(int(w) for w in news.get("windows", [3, 10, 20])),
        news_attn_z_window=int(news.get("attn_z_window", 252)),
        news_attn_shock_threshold=float(news.get("attn_shock_threshold", 2.0)),
        news_shock_windows=tuple(int(w) for w in news.get("shock_windows", [3, 10, 20])),
        news_compact_mode=bool(news.get("compact_mode", True)),
        news_topic_share_enabled=bool(news.get("topic_share_enabled", True)),
        news_include_roll_sum=bool(news.get("include_roll_sum", False)),
        news_include_roll_mean=bool(news.get("include_roll_mean", False)),
        news_include_roll_std=bool(news.get("include_roll_std", False)),
        news_include_tone_std=bool(news.get("include_tone_std", False)),
        news_include_tone_neg_share=bool(news.get("include_tone_neg_share", False)),
        news_include_interactions=bool(news.get("include_interactions", True)),
        news_health_min_nonzero_rate=float(health.get("min_nonzero_rate", 0.01)),
        news_health_min_unique_values=int(health.get("min_unique_values", 3)),
        news_health_drop_dead_queries=bool(health.get("drop_dead_queries", True)),
        news_health_fail_if_all_dead=bool(health.get("fail_if_all_dead", True)),
        news_query_ids=tuple(str(q) for q in news.get("topics_enabled", []) if str(q).strip()),
    )


# ---------------------------------------------------------------------------
# Per-asset feature builder (no cross-asset block)
# ---------------------------------------------------------------------------

def build_features_for_asset(
    asset_id: str,
    source_ticker: str,
    ingest_result: "IngestResult",
    features_config: str,
    db_path: str,
    feature_contract: list[str] | None = None,
) -> FeaturesResult:
    """Build the production feature vector for one asset on its data cutoff date.

    Adapts src/quant_risk/features/build.py for single-ticker production use.
    The cross-asset section is explicitly excluded (rpi5.md §7).

    Parameters
    ----------
    asset_id:
        Semantic asset identifier (used in log messages only).
    source_ticker:
        Raw ticker symbol to look up in raw_prices.
    ingest_result:
        Output of ingest_for_asset; provides data_cutoff_date.
    features_config:
        Path to config/prod/features.rpi5.yaml.
    db_path:
        Path to financial_data.duckdb.
    feature_contract:
        Optional list of expected feature column names from feature_contract.json.
        When supplied, the output DataFrame is filtered to these columns and
        missing_columns is populated for any that could not be produced.
    """
    forecast_date: date = ingest_result.data_cutoff_date
    forecast_ts = pd.Timestamp(forecast_date)

    yaml_cfg = load_yaml(features_config)
    cfg = _build_features_cfg(yaml_cfg, db_path)

    logger.debug(
        "[%s] Building features for forecast_date=%s ticker=%s",
        asset_id, forecast_date, source_ticker,
    )

    con = duckdb.connect(db_path, read_only=True)
    try:
        # ------------------------------------------------------------------ #
        # 1. Load prices (single ticker only)                                 #
        # ------------------------------------------------------------------ #
        dfp = con.execute(
            "SELECT ticker, date, close, volume "
            "FROM raw_prices WHERE ticker = ? ORDER BY date",
            [source_ticker],
        ).df()

        if dfp.empty:
            raise RuntimeError(
                f"[{asset_id}] No price data for ticker {source_ticker!r} in raw_prices."
            )
        dfp["date"] = pd.to_datetime(dfp["date"])

        # ------------------------------------------------------------------ #
        # 2. Load macro                                                        #
        # ------------------------------------------------------------------ #
        dfm = con.execute("SELECT * FROM macro_features ORDER BY date").df()
        if dfm.empty:
            raise RuntimeError(f"[{asset_id}] No data in macro_features.")
        dfm["date"] = pd.to_datetime(dfm["date"])

        # ------------------------------------------------------------------ #
        # 3. Build business-day master index up to forecast_date              #
        # ------------------------------------------------------------------ #
        start = max(dfp["date"].min(), dfm["date"].min())
        end = min(
            min(dfp["date"].max(), dfm["date"].max()),
            forecast_ts,
        )
        master_idx = pd.date_range(start=start, end=end, freq=cfg.calendar_freq)

        if forecast_ts not in master_idx:
            raise RuntimeError(
                f"[{asset_id}] forecast_date={forecast_date} not in business-day calendar "
                f"(range {master_idx[0].date()} – {master_idx[-1].date()})."
            )

        # ------------------------------------------------------------------ #
        # 4. Macro transforms + publication lags                              #
        # ------------------------------------------------------------------ #
        dfm2 = dfm.set_index("date").reindex(master_idx).ffill()
        dfm2.index.name = "date"
        dfm2 = dfm2.reset_index()

        macro_cols = ["vix", "fed_assets", "tga", "rrp", "m2", "ffr", "sofr", "net_liquidity"]
        macro_cols = [c for c in macro_cols if c in dfm2.columns]
        publication_lags = cfg.macro_publication_lags or {
            "vix": 0, "fed_assets": 1, "tga": 1, "rrp": 1,
            "m2": 5, "ffr": 1, "sofr": 1, "net_liquidity": 1,
        }
        for c in macro_cols:
            lag_bdays = int(publication_lags.get(c, 0))
            if lag_bdays > 0:
                dfm2[c] = dfm2[c].shift(lag_bdays)
        for c in macro_cols:
            dfm2[f"{c}_{cfg.macro_transform}"] = _macro_transform(
                dfm2[c].astype(float), cfg.macro_transform
            )
        for c in macro_cols:
            tc = f"{c}_{cfg.macro_transform}"
            for lag in cfg.macro_lags:
                dfm2[f"{tc}_lag{lag}"] = dfm2[tc].shift(lag)

        # ------------------------------------------------------------------ #
        # 5. News block (shared across tickers — here used for single asset)  #
        # ------------------------------------------------------------------ #
        news_block = _load_news_block(con, cfg, master_idx)

        # ------------------------------------------------------------------ #
        # 6. Per-asset price features                                         #
        # ------------------------------------------------------------------ #
        sub = dfp[dfp["ticker"] == source_ticker].copy()
        sub = sub.set_index("date").reindex(master_idx).ffill()
        sub.index.name = "date"
        sub = sub.reset_index()
        sub["ticker"] = source_ticker

        sub["logret"] = np.log(sub["close"]).diff()
        for lag in cfg.return_lags:
            sub[f"logret_lag{lag}"] = sub["logret"].shift(lag)

        rv_windows = {int(w) for w in cfg.rv_windows}
        rv_windows.update({20, int(cfg.rv_long_window)})
        for w in sorted(rv_windows):
            sub[f"rv_{w}"] = sub["logret"].rolling(w).std()

        sub[f"vol_{cfg.rv_long_window}"] = sub[f"rv_{cfg.rv_long_window}"]
        sub["rv_slope_20_60"] = sub["rv_20"] - sub[f"rv_{cfg.rv_long_window}"]
        sub["rv_ratio_20_60"] = sub["rv_20"] / sub[f"rv_{cfg.rv_long_window}"].replace(0.0, np.nan)
        sub["rv_logdiff_20_60"] = (
            np.log(sub["rv_20"].clip(lower=1e-12))
            - np.log(sub[f"rv_{cfg.rv_long_window}"].clip(lower=1e-12))
        )
        sub["vol_slope_20_60"] = sub["rv_slope_20_60"]
        sub["vol_ratio_20_60"] = sub["rv_ratio_20_60"]
        sub["vol_logdiff_20_60"] = sub["rv_logdiff_20_60"]

        ema_fast, ema_slow = cfg.rv_ema_spans
        rv20_ema_fast = sub["rv_20"].ewm(span=ema_fast, adjust=False).mean()
        rv20_ema_slow = sub["rv_20"].ewm(span=ema_slow, adjust=False).mean()
        sub[f"rv20_ema_diff_{ema_fast}_{ema_slow}"] = rv20_ema_fast - rv20_ema_slow

        sub["vol_of_vol_20"] = sub["rv_20"].rolling(cfg.vol_of_vol_window).std()
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

        log_volume = np.log(sub["volume"].clip(lower=1.0))
        vol_mu = log_volume.rolling(cfg.volume_z_window).mean()
        vol_sd = log_volume.rolling(cfg.volume_z_window).std().replace(0.0, np.nan)
        sub[f"volume_z_w{cfg.volume_z_window}"] = (log_volume - vol_mu) / vol_sd
        sub[f"volume_z_abs_w{cfg.volume_z_window}"] = sub[f"volume_z_w{cfg.volume_z_window}"].abs()

        # ------------------------------------------------------------------ #
        # 7. Merge macro + news                                               #
        # ------------------------------------------------------------------ #
        merged = sub.merge(dfm2, on="date", how="left")

        if cfg.news_enabled:
            merged = merged.merge(news_block, on="date", how="left")
            if bool(cfg.news_include_interactions):
                attn_cols = [
                    c for c in merged.columns
                    if c == "attn_z" or c.startswith("attn_z_")
                ]
                sigma_col = None
                for sigma_candidate in ("garch_sigma_t", "sigma_t"):
                    if sigma_candidate in merged.columns:
                        sigma_col = sigma_candidate
                        break
                for attn_col in attn_cols:
                    suffix = "" if attn_col == "attn_z" else attn_col[len("attn_z"):]
                    merged[f"attn_z_x_rv20{suffix}"] = merged[attn_col] * merged["rv_20"]
                    if "rv_5" in merged.columns:
                        merged[f"attn_z_x_rv5{suffix}"] = merged[attn_col] * merged["rv_5"]
                    if sigma_col is not None:
                        merged[f"attn_z_x_garch_sigma_t{suffix}"] = (
                            merged[attn_col] * merged[sigma_col]
                        )
                    tone_chg_col = f"tone_change_1d{suffix}"
                    if tone_chg_col in merged.columns:
                        merged[f"tone_change_1d_x_attn_z{suffix}"] = (
                            merged[tone_chg_col] * merged[attn_col]
                        )
                        merged[f"tone_change_1d_x_rv20{suffix}"] = (
                            merged[tone_chg_col] * merged["rv_20"]
                        )
                    tone_mean_col = f"tone_mean{suffix}"
                    if tone_mean_col in merged.columns:
                        merged[f"abs_tone_mean_x_attn_z{suffix}"] = (
                            merged[tone_mean_col].abs() * merged[attn_col]
                        )
                    tone_neg_col = f"tone_neg_share{suffix}"
                    if sigma_col is not None and tone_neg_col in merged.columns:
                        merged[f"tone_neg_share_x_garch_sigma_t{suffix}"] = (
                            merged[tone_neg_col] * merged[sigma_col]
                        )
                topic_cols = [
                    c for c in merged.columns
                    if c == "topic_share_z" or c.startswith("topic_share_z_")
                ]
                for topic_col in topic_cols:
                    suffix = (
                        "" if topic_col == "topic_share_z"
                        else topic_col[len("topic_share_z"):]
                    )
                    merged[f"topic_share_z_x_rv20{suffix}"] = merged[topic_col] * merged["rv_20"]
                    if "rv_5" in merged.columns:
                        merged[f"topic_share_z_x_rv5{suffix}"] = merged[topic_col] * merged["rv_5"]
                    tone_chg_col = f"tone_change_1d{suffix}"
                    if tone_chg_col in merged.columns:
                        merged[f"tone_change_1d_x_topic_share_z{suffix}"] = (
                            merged[tone_chg_col] * merged[topic_col]
                        )

        # ------------------------------------------------------------------ #
        # 8. Extract single row for forecast_date                             #
        # ------------------------------------------------------------------ #
        merged["date"] = pd.to_datetime(merged["date"])
        row = merged[merged["date"] == forecast_ts]

        if row.empty:
            raise RuntimeError(
                f"[{asset_id}] No feature row for forecast_date={forecast_date} "
                f"(ticker={source_ticker!r}). Data coverage may be insufficient."
            )

        # Drop raw price columns that are not features
        _non_feature_cols = {"ticker", "close", "volume"}
        feature_df = row.drop(
            columns=[c for c in _non_feature_cols if c in row.columns],
            errors="ignore",
        )
        feature_df = feature_df.set_index("date")

        # ------------------------------------------------------------------ #
        # 9. Validate / filter against feature_contract                       #
        # ------------------------------------------------------------------ #
        missing_columns: list[str] = []
        if feature_contract:
            missing_columns = [c for c in feature_contract if c not in feature_df.columns]
            available = [c for c in feature_contract if c in feature_df.columns]
            feature_df = feature_df[available]

        logger.debug(
            "[%s] Features built — %d columns, %d missing",
            asset_id, len(feature_df.columns), len(missing_columns),
        )
        if missing_columns:
            logger.warning(
                "[%s] %d feature(s) missing from contract: %s",
                asset_id, len(missing_columns), missing_columns[:10],
            )

        return FeaturesResult(
            features=feature_df,
            feature_count=len(feature_df.columns),
            feature_date=forecast_date,
            missing_columns=missing_columns,
        )

    finally:
        con.close()
