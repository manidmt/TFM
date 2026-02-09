'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-24

@description: Module to build datasets for model training/evaluation.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import duckdb
import pandas as pd
import numpy as np

from quant_risk.models.econometric.sarimax import SarimaxConfig, fit_sarimax, make_sarimax_features


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
        # drop target-like realized vol features
        if c.startswith("rv_"):
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
    con = duckdb.connect(cfg.db_path)
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
    if per_ticker:
        def bin_row(row):
            cuts = bins.loc[row["ticker"]].values
            return int(np.digitize(row["vol_fwd"], cuts, right=True))
        return df.apply(bin_row, axis=1)
    else:
        return pd.Series(np.digitize(df["vol_fwd"].values, bins, right=True), index=df.index).astype(int)



def build_xy(df: pd.DataFrame, feature_cols: Sequence[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[list(feature_cols)].copy()
    y = df["regime"].copy()
    return X, y


def make_dataset(cfg: DatasetConfig) -> dict:
    df = load_joined(cfg)

    if cfg.pooled:
        df = pd.get_dummies(df, columns=["ticker"], prefix="ticker")

    feature_cols = _infer_feature_columns(df)
    assert "vol_fwd" not in feature_cols
    feature_cols = [c for c in feature_cols if c != "date"]
    df_model = df.dropna(subset=feature_cols + ["vol_fwd"]).copy()

    train, valid, test = make_splits(df_model, cfg.train_end, cfg.valid_end, cfg.horizon)

    bins = _fit_bins(train, regime_bins=cfg.regime_bins, per_ticker=True)
    for split in (train, valid, test):
        split["regime"] = _apply_bins(
            split, bins, regime_bins=cfg.regime_bins, per_ticker=True
        )

    # After computing regime bins on train and before dropna final
    # Insert SARIMAX feature generation if enabled

    if cfg.use_sarimax:
        print("Fitting SARIMAX models on training data...")

        sarimax_cfg = SarimaxConfig(
            order=cfg.sarimax_order,
            seasonal_order=cfg.sarimax_seasonal_order,
            trend=cfg.sarimax_trend,
            horizon=cfg.horizon,
            target_col="vol_fwd",
            log_transform=cfg.sarimax_log_transform,
            exog_cols=cfg.sarimax_exog_cols,
            train_nobs_by_ticker=train.groupby("ticker").size().to_dict(),
        )

        train_for_sarimax = train[["ticker", "date", "vol_fwd"] + list(cfg.sarimax_exog_cols)].copy()
        fitted_models = fit_sarimax(train_for_sarimax, sarimax_cfg)
        
        print("Generating SARIMAX features for all rows (train/valid/test) ...")
        sarimax_cols = [
            f"sarimax_fcst_mean_h{cfg.horizon}",
            f"sarimax_fcst_se_h{cfg.horizon}",
            f"sarimax_ci_width_h{cfg.horizon}",
            "sarimax_resid",
            "sarimax_resid_std",
        ]

        combined = pd.concat([train, valid, test], ignore_index=True).sort_values(["ticker", "date"])
        combined = make_sarimax_features(combined, fitted_models, sarimax_cfg)

        missing = [c for c in sarimax_cols if c not in combined.columns]
        if missing:
            raise RuntimeError(f"SARIMAX columns not present after feature generation: {missing}")

        for c in sarimax_cols:
            if c not in feature_cols:
                feature_cols.append(c)

        df_model = combined.copy()
    else:
        df_model = pd.concat([train, valid, test], ignore_index=True).sort_values(["ticker", "date"])
    
    df_model = df_model.dropna(subset=feature_cols + ["vol_fwd"]).copy()

    train, valid, test = make_splits(df_model, cfg.train_end, cfg.valid_end, cfg.horizon)

    bins = _fit_bins(train, regime_bins=cfg.regime_bins, per_ticker=True)
    for split in (train, valid, test):
        split["regime"] = _apply_bins(
            split, bins, regime_bins=cfg.regime_bins, per_ticker=True
        )

    # After bins applied
    df_binned = pd.concat([train, valid, test], ignore_index=True).sort_values(["ticker", "date"])

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