'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-23

@description: Focused tests for GARCH econometric feature alignment and leakage safety.
'''

import numpy as np
import pandas as pd

from quant_risk.models.econometric.garch import GarchConfig, fit_garch, make_garch_features


def _simulate_returns(n: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    sigma2 = np.full(n, 1e-4, dtype=float)
    ret = np.zeros(n, dtype=float)

    omega = 1e-6
    alpha = 0.07
    beta = 0.91

    for t in range(1, n):
        sigma2[t] = omega + alpha * (ret[t - 1] ** 2) + beta * sigma2[t - 1]
        ret[t] = np.sqrt(max(sigma2[t], 1e-12)) * eps[t]

    return ret


def _sample_df() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=180, freq="B")
    parts = []
    for idx, ticker in enumerate(["A", "B"]):
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "logret": _simulate_returns(len(dates), seed=200 + idx),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_garch_features_later_rows_non_nan():
    full_df = _sample_df()
    train_df = full_df[full_df["date"] <= pd.Timestamp("2021-06-30")].copy()

    cfg = GarchConfig(horizon=5)
    fitted = fit_garch(train_df, cfg)
    out = make_garch_features(full_df, fitted, cfg)

    fwd_col = "garch_sigma_fwd_h5"
    ticker_a = out[out["ticker"] == "A"].sort_values("date")
    assert ticker_a.tail(30)[fwd_col].notna().sum() >= 15


def test_garch_features_no_future_perturbation_changes_past():
    horizon = 5
    cutoff = pd.Timestamp("2021-05-31")

    base = _sample_df().sort_values(["ticker", "date"]).copy()
    train_df = base[base["date"] <= cutoff].copy()

    cfg = GarchConfig(horizon=horizon)
    fitted = fit_garch(train_df, cfg)

    altered = base.copy()
    altered.loc[altered["date"] > pd.Timestamp("2021-07-15"), "logret"] *= 20.0

    out_base = make_garch_features(base, fitted, cfg)
    out_alt = make_garch_features(altered, fitted, cfg)

    fwd_col = f"garch_sigma_fwd_h{horizon}"
    pre_cutoff = out_base["date"] <= cutoff
    diff = (out_base.loc[pre_cutoff, fwd_col] - out_alt.loc[pre_cutoff, fwd_col]).abs()

    assert diff.fillna(0.0).max() < 1e-12


def test_garch_features_alignment_sanity():
    horizon = 5
    full_df = _sample_df().sort_values(["ticker", "date"]).copy()
    train_df = full_df[full_df["date"] <= pd.Timestamp("2021-06-30")].copy()

    cfg = GarchConfig(horizon=horizon)
    fitted = fit_garch(train_df, cfg)
    out = make_garch_features(full_df, fitted, cfg)

    fwd_col = f"garch_sigma_fwd_h{horizon}"
    ticker_a = out[out["ticker"] == "A"].sort_values("date").copy()

    corr_now = pd.concat(
        [ticker_a[fwd_col], ticker_a["logret"].abs()], axis=1
    ).dropna().corr().iloc[0, 1]
    corr_fwd = pd.concat(
        [ticker_a[fwd_col], ticker_a["logret"].abs().shift(-horizon)], axis=1
    ).dropna().corr().iloc[0, 1]

    assert np.isfinite(corr_now)
    assert np.isfinite(corr_fwd)
    assert corr_fwd > -0.05
    assert corr_fwd >= (corr_now - 0.20)
