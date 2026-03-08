'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-27

@description: Tests for chained SARIMAX->GARCH dataset generation.
'''

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from quant_risk.datasets.make_dataset import DatasetConfig, make_dataset


DB = "data/db/financial_data.duckdb"


def _make_synthetic_panel(horizon: int = 5, n_dates: int = 260):
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2018-01-01", periods=n_dates)
    tickers = ["AAA", "BBB"]

    feature_rows = []
    label_rows = []
    for ticker in tickers:
        eps = rng.normal(0.0, 0.01, size=n_dates)
        logret = np.zeros(n_dates, dtype=float)
        for i in range(1, n_dates):
            logret[i] = 0.15 * logret[i - 1] + eps[i]

        macro = rng.normal(0.0, 1.0, size=n_dates).cumsum()
        macro_diff = np.concatenate([[0.0], np.diff(macro)])
        lag1 = np.concatenate([[0.0], logret[:-1]])
        lag5 = np.concatenate([np.zeros(5), logret[:-5]])

        vol_fwd = np.full(n_dates, np.nan, dtype=float)
        for i in range(n_dates - horizon):
            vol_fwd[i] = float(np.std(logret[i + 1 : i + 1 + horizon], ddof=0))

        for i, d in enumerate(dates):
            feature_rows.append(
                {
                    "ticker": ticker,
                    "date": d,
                    "logret": float(logret[i]),
                    "logret_lag1": float(lag1[i]),
                    "logret_lag5": float(lag5[i]),
                    "net_liquidity_diff": float(macro_diff[i]),
                    "macro_x": float(macro[i]),
                }
            )
            label_rows.append(
                {
                    "ticker": ticker,
                    "date": d,
                    "horizon": int(horizon),
                    "vol_fwd": float(vol_fwd[i]) if np.isfinite(vol_fwd[i]) else np.nan,
                }
            )

    df_feat = pd.DataFrame(feature_rows)
    df_lab = pd.DataFrame(label_rows).dropna(subset=["vol_fwd"]).copy()
    return df_feat, df_lab, dates


def _write_db(path: Path, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE OR REPLACE TABLE features_daily AS SELECT * FROM features_df")
        con.execute("CREATE OR REPLACE TABLE labels_regime AS SELECT * FROM labels_df")
    finally:
        con.close()


def test_chain_features_and_anti_leakage(tmp_path: Path):
    horizon = 5
    features_base, labels, dates = _make_synthetic_panel(horizon=horizon)
    cutoff = pd.Timestamp(dates[180])

    features_perturbed = features_base.copy()
    mask_future = features_perturbed["date"] > cutoff
    features_perturbed.loc[mask_future, "logret"] = (
        features_perturbed.loc[mask_future, "logret"] + 0.1
    )
    features_perturbed["logret_lag1"] = features_perturbed.groupby("ticker")["logret"].shift(1).fillna(0.0)
    features_perturbed["logret_lag5"] = features_perturbed.groupby("ticker")["logret"].shift(5).fillna(0.0)

    db_a = tmp_path / "chain_a.duckdb"
    db_b = tmp_path / "chain_b.duckdb"
    _write_db(db_a, features_base, labels)
    _write_db(db_b, features_perturbed, labels)

    cfg_kwargs = dict(
        features_table="features_daily",
        labels_table="labels_regime",
        tickers=("AAA", "BBB"),
        horizon=horizon,
        train_end=str(dates[140].date()),
        valid_end=str(dates[200].date()),
        pooled=False,
        regime_bins=3,
        use_sarimax_garch_chain=True,
        sarimax_chain_exog_cols=(),
        garch_p=1,
        garch_q=1,
    )
    pack_a = make_dataset(DatasetConfig(db_path=str(db_a), **cfg_kwargs))
    pack_b = make_dataset(DatasetConfig(db_path=str(db_b), **cfg_kwargs))

    assert "sarimax_resid" in pack_a["df"].columns
    assert f"garch_sigma_fwd_h{horizon}" in pack_a["df"].columns
    assert pack_a["df"]["sarimax_resid"].notna().sum() > 0
    assert pack_a["df"][f"garch_sigma_fwd_h{horizon}"].notna().sum() > 0

    cols = ["sarimax_resid", f"garch_sigma_fwd_h{horizon}", f"garch_var_fwd_h{horizon}"]
    left = pack_a["df"][["ticker", "date"] + cols].copy()
    right = pack_b["df"][["ticker", "date"] + cols].copy()
    merged = left.merge(right, on=["ticker", "date"], suffixes=("_a", "_b"), how="inner")
    merged = merged[merged["date"] <= cutoff].copy()

    for c in cols:
        a = merged[f"{c}_a"].to_numpy(dtype=float)
        b = merged[f"{c}_b"].to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        assert mask.sum() > 20
        max_diff = float(np.max(np.abs(a[mask] - b[mask])))
        assert max_diff < 1e-12


def test_chain_make_dataset_real_smoke():
    cfg = DatasetConfig(
        db_path=DB,
        tickers=("^GSPC",),
        horizon=20,
        pooled=False,
        use_sarimax_garch_chain=True,
        sarimax_chain_exog_cols=(),
    )
    pack = make_dataset(cfg)

    assert len(pack["train"]) > 0
    assert len(pack["valid"]) > 0
    assert len(pack["test"]) > 0
    assert any(c.startswith("sarimax_") for c in pack["feature_cols"])
    assert any(c.startswith("garch_") for c in pack["feature_cols"])


def test_chain_har_make_dataset_real_smoke():
    cfg = DatasetConfig(
        db_path=DB,
        tickers=("^GSPC",),
        horizon=20,
        pooled=False,
        use_sarimax_garch_chain=True,
        chain_mean_model="har",
        har_target_col="rv_20",
        har_exog_cols=(),
        garch_vol="Garch",
    )
    pack = make_dataset(cfg)

    assert len(pack["train"]) > 0
    assert len(pack["valid"]) > 0
    assert len(pack["test"]) > 0
    assert any(c.startswith("har_") for c in pack["feature_cols"])
    assert "sigma_diff1" in pack["feature_cols"]
    assert "sigma_diff2" in pack["feature_cols"]
    assert "abs_std_resid" in pack["feature_cols"]
    assert "regime_boundary_distance" in pack["feature_cols"]
