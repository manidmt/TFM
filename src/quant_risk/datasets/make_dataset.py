'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-24

@description: Module to build datasets for model training/evaluation.
'''

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence
import re

import duckdb
import pandas as pd
import numpy as np

from quant_risk.models.econometric.sarimax import SarimaxConfig, fit_sarimax, make_sarimax_features
from quant_risk.models.econometric.garch import GarchConfig, fit_garch, make_garch_features
from quant_risk.models.econometric.egarch import EgarchConfig, fit_egarch, make_egarch_features
from quant_risk.models.econometric.gjrgarch import (
    GjrGarchConfig,
    fit_gjrgarch,
    make_gjrgarch_features,
)


@dataclass(frozen=True)
class DatasetConfig:
    db_path: str
    features_table: str = "features_daily"
    labels_table: str = "labels_regime"

    tickers: tuple[str, ...] = ("^GSPC", "BTC-USD", "TLT")
    horizon: int = 20

    # split by date (time-series split)
    train_end: str | None = None
    valid_end: str | None = None 
    # test is (valid_end, max]

    pooled: bool = False
    regime_bins: int = 3
    use_sarimax: bool = False
    sarimax_order: tuple[int, int, int] = (1, 0, 1)
    sarimax_seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    sarimax_trend: str | None = "c"
    sarimax_log_transform: bool = True
    sarimax_exog_cols: tuple[str, ...] = ()
    sarimax_target_col: str = "rv_20"
    use_sarimax_garch_chain: bool = False
    sarimax_chain_exog_cols: tuple[str, ...] = ()
    use_garch: bool = False
    garch_p: int = 1
    garch_o: int = 1
    garch_q: int = 1
    garch_dist: str = "normal"
    garch_mean: str = "zero"
    garch_vol: str = "Garch"
    garch_target_col: str = "logret"
    garch_agg: str = "last"
    garch_chain_agg: str = "rms"
    garch_annualize: bool = False
    garch_scale: float = 100.0

def _infer_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select model features from features_daily.
    Excludes identifiers and raw-ish columns that can leak or be redundant.
    """
    exclude = {
        "date",
        "regime",
        "source",
        "inserted_at",
        "close",
        "volume",
        "vol_fwd",
    }

    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        # Drop only raw realized-vol columns (rv_5, rv_20, rv_60, ...).
        # Keep engineered volatility features (e.g., rv_slope_20_60, rv_ratio_20_60)
        # and non-rv prefixed aliases (vol_*).
        if re.fullmatch(r"rv_\d+", c):
            continue
        if c == "logret":
            continue
        cols.append(c)

    # Keep only numeric
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    keep = []
    for c in num_cols:
        if df[c].notna().mean() >= 0.98:
            keep.append(c)

    return keep


def load_joined(cfg: DatasetConfig) -> pd.DataFrame:
    # Read-only connection avoids cross-process file locks during parallel runs.
    con = duckdb.connect(cfg.db_path, read_only=True)
    try:
        # features
        df_feat = con.execute(
            f"""
            SELECT * FROM {cfg.features_table}
            WHERE ticker IN ({",".join(["?"] * len(cfg.tickers))})
            ORDER BY ticker, date
            """,
            list(cfg.tickers),
        ).df()

        if df_feat.empty:
            raise RuntimeError("features_daily empty for selected tickers.")

        # labels
        df_lab = con.execute(
            f"""
            SELECT ticker, date, horizon, vol_fwd
            FROM {cfg.labels_table}
            WHERE horizon = ?
            AND ticker IN ({",".join(["?"] * len(cfg.tickers))})
            ORDER BY ticker, date
            """,
            [cfg.horizon, *cfg.tickers],
        ).df()

        if df_lab.empty:
            raise RuntimeError("labels_regime empty for selected tickers/horizon.")

    finally:
        con.close()

    df_feat["date"] = pd.to_datetime(df_feat["date"])
    df_lab["date"] = pd.to_datetime(df_lab["date"])

    df = df_feat.merge(df_lab[["ticker", "date", "vol_fwd"]], on=["ticker", "date"], how="inner")

    return df


def make_splits(df: pd.DataFrame, train_end: str | None, valid_end: str | None, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-series split by absolute dates.
    If train_end/valid_end not provided, infer by quantiles on the pooled timeline.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    if train_end is None or valid_end is None:
        dates = df["date"].sort_values().unique()
        n = len(dates)
        train_end_dt = dates[int(n * 0.70)]
        valid_end_dt = dates[int(n * 0.85)]
    else:
        train_end_dt = pd.to_datetime(train_end)
        valid_end_dt = pd.to_datetime(valid_end)

    valid_start = train_end_dt + pd.offsets.BDay(horizon)
    test_start  = valid_end_dt + pd.offsets.BDay(horizon)

    train = df[df["date"] <= train_end_dt].copy()
    valid = df[(df["date"] >= valid_start) & (df["date"] <= valid_end_dt)].copy()
    test  = df[df["date"] >= test_start].copy()

    return train, valid, test

def _fit_bins(train: pd.DataFrame, regime_bins: int, per_ticker: bool = True):
    if per_ticker:
        qs = train.groupby("ticker")["vol_fwd"].quantile([i/regime_bins for i in range(1, regime_bins)]).unstack()
        # qs[ticker] = [q1, q2] si 3 bins
        return qs
    else:
        cuts = train["vol_fwd"].quantile([i/regime_bins for i in range(1, regime_bins)]).tolist()
        return cuts

def _apply_bins(df: pd.DataFrame, bins, regime_bins: int, per_ticker: bool = True):
    if df.empty:
        return pd.Series(index=df.index, dtype=int)
    if per_ticker:
        def bin_row(row):
            cuts = bins.loc[row["ticker"]].values
            return int(np.digitize(row["vol_fwd"], cuts, right=True))
        return df.apply(bin_row, axis=1)
    else:
        return pd.Series(np.digitize(df["vol_fwd"].values, bins, right=True), index=df.index).astype(int)


def _assert_no_future_cols(cols: tuple[str, ...], banned: set[str], context: str) -> None:
    bad = [c for c in cols if c in banned]
    if bad:
        raise ValueError(
            f"{context}: forbidden columns for dataset: {bad}. "
            f"Banned={sorted(banned)}"
        )



def build_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[list(feature_cols)].copy()
    y = df["regime"].astype(int).copy()
    return X, y


def _vol_prefix_from_kind(vol_kind: str) -> str:
    vk = str(vol_kind).lower()
    if vk == "garch":
        return "garch"
    if vk in {"egarch", "e-garch"}:
        return "egarch"
    if vk in {"gjr-garch", "gjrgarch", "gjr"}:
        return "gjrgarch"
    raise ValueError(
        f"Unsupported garch_vol='{vol_kind}'. Use 'Garch', 'EGARCH' or 'GJR-GARCH'."
    )


def _vol_feature_columns(prefix: str, horizon: int) -> list[str]:
    return [
        f"{prefix}_sigma_t",
        f"{prefix}_sigma_fwd_h{horizon}",
        f"{prefix}_var_fwd_h{horizon}",
        f"{prefix}_var_mean_h{horizon}",
        f"{prefix}_sigma_rms_h{horizon}",
        f"{prefix}_resid",
        f"{prefix}_z",
        f"{prefix}_delta_sigma_h{horizon}",
        f"{prefix}_ratio_sigma_h{horizon}",
        f"{prefix}_sigma_t_diff1",
        f"{prefix}_sigma_t_diff2",
    ]


def make_dataset(cfg: DatasetConfig) -> dict:
    df = load_joined(cfg)

    if cfg.use_sarimax and cfg.sarimax_target_col == "vol_fwd":
        raise ValueError(
            "sarimax_target_col='vol_fwd' would leak future information. "
            "Use a past-only series such as rv_20."
        )
    if cfg.use_garch and cfg.garch_target_col == "vol_fwd":
        raise ValueError(
            "garch_target_col='vol_fwd' would leak future information. "
            "Use a past-only return/vol proxy such as logret or rv_20."
        )
    if cfg.use_sarimax_garch_chain and (cfg.use_sarimax or cfg.use_garch):
        raise ValueError(
            "use_sarimax_garch_chain=True is mutually exclusive with use_sarimax/use_garch. "
            "The chain mode already computes both stages."
        )
    banned_future = {"vol_fwd"}
    _assert_no_future_cols(
        tuple(cfg.sarimax_exog_cols), banned_future, "sarimax_exog_cols"
    )
    _assert_no_future_cols(
        tuple(cfg.sarimax_chain_exog_cols), banned_future, "sarimax_chain_exog_cols"
    )

    if cfg.pooled:
        ticker_dummies = pd.get_dummies(df["ticker"], prefix="ticker")
        df = pd.concat([df, ticker_dummies], axis=1)

    # Infer base feature columns from TRAIN only to avoid selection leakage.
    train_for_feature_selection, _, _ = make_splits(df, cfg.train_end, cfg.valid_end, cfg.horizon)
    if train_for_feature_selection.empty:
        raise RuntimeError("Training split is empty; cannot infer feature columns.")

    feature_cols = _infer_feature_columns(train_for_feature_selection)
    assert "vol_fwd" not in feature_cols
    feature_cols = [c for c in feature_cols if c != "date"]
    df_model = df.dropna(subset=feature_cols + ["vol_fwd"]).copy()

    train, valid, test = make_splits(df_model, cfg.train_end, cfg.valid_end, cfg.horizon)

    bins = _fit_bins(train, regime_bins=cfg.regime_bins, per_ticker=True)
    for split in (train, valid, test):
        split["regime"] = _apply_bins(
            split, bins, regime_bins=cfg.regime_bins, per_ticker=True
        )

    # Build econometric features on the full chronological panel.
    # This preserves in-between gap dates in model state updates and keeps
    # walk-forward recursion realistic while remaining leakage-safe.
    combined = df_model.sort_values(["ticker", "date"]).copy()

    def _append_vol_dynamic_features(df_in: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add volatility transition features (delta/ratio/diff/acceleration)."""
        out_df = df_in.sort_values(["ticker", "date"]).copy()
        sigma_t_col = f"{prefix}_sigma_t"
        sigma_h_col = f"{prefix}_sigma_fwd_h{cfg.horizon}"
        if sigma_t_col not in out_df.columns or sigma_h_col not in out_df.columns:
            return out_df

        delta_col = f"{prefix}_delta_sigma_h{cfg.horizon}"
        ratio_col = f"{prefix}_ratio_sigma_h{cfg.horizon}"
        diff_col = f"{prefix}_sigma_t_diff1"
        accel_col = f"{prefix}_sigma_t_diff2"

        out_df[delta_col] = out_df[sigma_h_col] - out_df[sigma_t_col]
        out_df[ratio_col] = out_df[sigma_h_col] / out_df[sigma_t_col].replace(0.0, np.nan)
        out_df[diff_col] = out_df.groupby("ticker")[sigma_t_col].diff(1)
        out_df[accel_col] = out_df.groupby("ticker")[diff_col].diff(1)
        return out_df

    def _make_vol_objects(
        *,
        target_col: str,
        agg: str,
        train_sizes: dict[str, int],
    ) -> tuple[str, Any, Any, Any]:
        vol_prefix = _vol_prefix_from_kind(cfg.garch_vol)
        if vol_prefix == "garch":
            vol_cfg = GarchConfig(
                p=cfg.garch_p,
                q=cfg.garch_q,
                dist=cfg.garch_dist,
                mean=cfg.garch_mean,
                vol=cfg.garch_vol,
                horizon=cfg.horizon,
                target_col=target_col,
                agg=agg,
                annualize=cfg.garch_annualize,
                scale=cfg.garch_scale,
                train_nobs_by_ticker=train_sizes,
            )
            return vol_prefix, vol_cfg, fit_garch, make_garch_features
        if vol_prefix == "egarch":
            vol_cfg = EgarchConfig(
                p=cfg.garch_p,
                q=cfg.garch_q,
                dist=cfg.garch_dist,
                mean=cfg.garch_mean,
                vol=cfg.garch_vol,
                horizon=cfg.horizon,
                target_col=target_col,
                agg=agg,
                annualize=cfg.garch_annualize,
                scale=cfg.garch_scale,
                train_nobs_by_ticker=train_sizes,
            )
            return vol_prefix, vol_cfg, fit_egarch, make_egarch_features

        vol_cfg = GjrGarchConfig(
            p=cfg.garch_p,
            o=cfg.garch_o,
            q=cfg.garch_q,
            dist=cfg.garch_dist,
            mean=cfg.garch_mean,
            vol=cfg.garch_vol,
            horizon=cfg.horizon,
            target_col=target_col,
            agg=agg,
            annualize=cfg.garch_annualize,
            scale=cfg.garch_scale,
            train_nobs_by_ticker=train_sizes,
        )
        return vol_prefix, vol_cfg, fit_gjrgarch, make_gjrgarch_features

    if cfg.use_sarimax_garch_chain:
        print("Fitting chained SARIMAX->GARCH models on training data...")

        chain_exog_cols = (
            cfg.sarimax_chain_exog_cols
            if cfg.sarimax_chain_exog_cols
            else cfg.sarimax_exog_cols
        )

        sarimax_cfg_chain = SarimaxConfig(
            order=cfg.sarimax_order,
            seasonal_order=cfg.sarimax_seasonal_order,
            trend=cfg.sarimax_trend,
            horizon=cfg.horizon,
            target_col="logret",
            log_transform=False,
            exog_cols=chain_exog_cols,
            train_nobs_by_ticker=train.groupby("ticker").size().to_dict(),
        )

        train_for_sarimax_chain = train[
            ["ticker", "date", "logret"] + list(chain_exog_cols)
        ].copy()
        fitted_sarimax_chain = fit_sarimax(train_for_sarimax_chain, sarimax_cfg_chain)

        print("Generating chained SARIMAX features for all rows (train/valid/test) ...")
        combined = make_sarimax_features(combined, fitted_sarimax_chain, sarimax_cfg_chain)

        if "sarimax_resid" not in combined.columns:
            raise RuntimeError("SARIMAX chain failed: 'sarimax_resid' was not generated.")

        sarimax_cols = [
            f"sarimax_fcst_mean_h{cfg.horizon}",
            f"sarimax_fcst_se_h{cfg.horizon}",
            f"sarimax_ci_width_h{cfg.horizon}",
            "sarimax_resid",
            "sarimax_resid_std",
        ]
        for c in sarimax_cols:
            if c in combined.columns and c not in feature_cols:
                feature_cols.append(c)

        train_keys = train[["ticker", "date"]].drop_duplicates()
        train_enriched = combined.merge(train_keys, on=["ticker", "date"], how="inner")
        train_for_vol_chain = train_enriched[["ticker", "date", "sarimax_resid"]].dropna()
        vol_train_sizes = train_for_vol_chain.groupby("ticker").size().to_dict()
        vol_prefix, vol_cfg_chain, fit_vol, make_vol_features = _make_vol_objects(
            target_col="sarimax_resid",
            agg=cfg.garch_chain_agg,
            train_sizes=vol_train_sizes,
        )
        if getattr(vol_cfg_chain, "mean", "zero") != "zero":
            vol_cfg_chain = replace(vol_cfg_chain, mean="zero")

        fitted_vol_chain = fit_vol(train_for_vol_chain, vol_cfg_chain)

        print(f"Generating chained {cfg.garch_vol} features for all rows (train/valid/test) ...")
        combined = make_vol_features(combined, fitted_vol_chain, vol_cfg_chain)
        combined = _append_vol_dynamic_features(combined, prefix=vol_prefix)

        for c in _vol_feature_columns(vol_prefix, cfg.horizon):
            if c in combined.columns and c not in feature_cols:
                feature_cols.append(c)

    if cfg.use_sarimax and not cfg.use_sarimax_garch_chain:
        print("Fitting SARIMAX models on training data...")

        sarimax_cfg = SarimaxConfig(
            order=cfg.sarimax_order,
            seasonal_order=cfg.sarimax_seasonal_order,
            trend=cfg.sarimax_trend,
            horizon=cfg.horizon,
            target_col=cfg.sarimax_target_col,
            log_transform=cfg.sarimax_log_transform,
            exog_cols=cfg.sarimax_exog_cols,
            train_nobs_by_ticker=train.groupby("ticker").size().to_dict(),
        )

        train_for_sarimax = train[
            ["ticker", "date", cfg.sarimax_target_col] + list(cfg.sarimax_exog_cols)
        ].copy()
        fitted_models = fit_sarimax(train_for_sarimax, sarimax_cfg)

        print("Generating SARIMAX features for all rows (train/valid/test) ...")
        sarimax_cols = [
            f"sarimax_fcst_mean_h{cfg.horizon}",
            f"sarimax_fcst_se_h{cfg.horizon}",
            f"sarimax_ci_width_h{cfg.horizon}",
            "sarimax_resid",
            "sarimax_resid_std",
        ]

        combined = make_sarimax_features(combined, fitted_models, sarimax_cfg)

        missing = [c for c in sarimax_cols if c not in combined.columns]
        if missing:
            raise RuntimeError(f"SARIMAX columns not present after feature generation: {missing}")

        for c in sarimax_cols:
            if c not in feature_cols:
                feature_cols.append(c)

    if cfg.use_garch and not cfg.use_sarimax_garch_chain:
        print(f"Fitting {cfg.garch_vol} models on training data...")

        train_for_garch = train[["ticker", "date", cfg.garch_target_col]].copy()
        vol_prefix, vol_cfg, fit_vol, make_vol_features = _make_vol_objects(
            target_col=cfg.garch_target_col,
            agg=cfg.garch_agg,
            train_sizes=train.groupby("ticker").size().to_dict(),
        )
        fitted_garch = fit_vol(train_for_garch, vol_cfg)

        print(f"Generating {cfg.garch_vol} features for all rows (train/valid/test) ...")
        combined = make_vol_features(combined, fitted_garch, vol_cfg)
        combined = _append_vol_dynamic_features(combined, prefix=vol_prefix)

        missing = [
            c
            for c in [
                f"{vol_prefix}_sigma_t",
                f"{vol_prefix}_sigma_fwd_h{cfg.horizon}",
                f"{vol_prefix}_var_fwd_h{cfg.horizon}",
                f"{vol_prefix}_resid",
                f"{vol_prefix}_z",
            ]
            if c not in combined.columns
        ]
        if missing:
            raise RuntimeError(f"{cfg.garch_vol} columns not present after feature generation: {missing}")

        for c in _vol_feature_columns(vol_prefix, cfg.horizon):
            if c in combined.columns and c not in feature_cols:
                feature_cols.append(c)

    df_model = combined.copy()
    
    df_model = df_model.dropna(subset=feature_cols + ["vol_fwd"]).copy()

    train, valid, test = make_splits(df_model, cfg.train_end, cfg.valid_end, cfg.horizon)

    bins = _fit_bins(train, regime_bins=cfg.regime_bins, per_ticker=True)
    for split in (train, valid, test):
        split["regime"] = _apply_bins(
            split, bins, regime_bins=cfg.regime_bins, per_ticker=True
        )

    # After bins applied
    df_binned = pd.concat([train, valid, test], ignore_index=True).sort_values(["ticker", "date"])
    df_binned["regime"] = df_binned["regime"].astype(int)

    return {
        "df": df_binned,
        "feature_cols": feature_cols,
        "train": train,
        "valid": valid,
        "test": test,
        "bins": bins,
        "n_rows": len(df_binned),
        "n_features": len(feature_cols),
        "start": str(df_binned["date"].min().date()),
        "end": str(df_binned["date"].max().date()),
    }
