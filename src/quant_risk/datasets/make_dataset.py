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
            SELECT ticker, date, horizon, regime
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

    df = df_feat.merge(df_lab[["ticker", "date", "regime"]], on=["ticker", "date"], how="inner")

    df["regime"] = df["regime"].astype(int)

    return df


def make_splits(df: pd.DataFrame, train_end: str | None, valid_end: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    train = df[df["date"] <= train_end_dt].copy()
    valid = df[(df["date"] > train_end_dt) & (df["date"] <= valid_end_dt)].copy()
    test = df[df["date"] > valid_end_dt].copy()

    return train, valid, test


def build_xy(df: pd.DataFrame, feature_cols: Sequence[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[list(feature_cols)].copy()
    y = df["regime"].copy()
    return X, y


def make_dataset(cfg: DatasetConfig) -> dict:
    df = load_joined(cfg)

    if cfg.pooled:
        df = pd.get_dummies(df, columns=["ticker"], prefix="ticker")

    feature_cols = _infer_feature_columns(df)
    feature_cols = [c for c in feature_cols if c != "date"]
    df_model = df.dropna(subset=feature_cols + ["regime"]).copy()

    train, valid, test = make_splits(df_model, cfg.train_end, cfg.valid_end)

    return {
        "df": df_model,
        "feature_cols": feature_cols,
        "train": train,
        "valid": valid,
        "test": test,
        "n_rows": len(df_model),
        "n_features": len(feature_cols),
        "start": str(df_model["date"].min().date()),
        "end": str(df_model["date"].max().date()),
    }