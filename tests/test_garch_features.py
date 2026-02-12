'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-12

@description: Tests for GARCH-based volatility features.
'''

import numpy as np
import pandas as pd
import pytest

from quant_risk.models.econometric.garch import GarchConfig, fit_garch, make_garch_features


def _simulate_garch_like_returns(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    sigma2 = np.full(n, 1e-4, dtype=float)
    ret = np.zeros(n, dtype=float)

    omega = 1e-6
    alpha = 0.08
    beta = 0.90

    for t in range(1, n):
        sigma2[t] = omega + alpha * (ret[t - 1] ** 2) + beta * sigma2[t - 1]
        ret[t] = np.sqrt(max(sigma2[t], 1e-12)) * eps[t]

    return ret


@pytest.fixture
def sample_garch_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=220, freq="B")

    parts = []
    for i, ticker in enumerate(["A", "B"]):
        logret = _simulate_garch_like_returns(len(dates), seed=100 + i)
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "logret": logret,
                }
            )
        )

    return pd.concat(parts, ignore_index=True)


def test_fit_garch_returns_models_per_ticker(sample_garch_data):
    train_df = sample_garch_data[sample_garch_data["date"] <= pd.Timestamp("2020-09-30")].copy()
    cfg = GarchConfig(horizon=5)

    fitted = fit_garch(train_df, cfg)

    assert "A" in fitted
    assert "B" in fitted
    assert fitted["A"] is not None
    assert fitted["B"] is not None
    assert "params" in fitted["A"]
    assert "params" in fitted["B"]


def test_make_garch_features_later_rows_have_values(sample_garch_data):
    train_df = sample_garch_data[sample_garch_data["date"] <= pd.Timestamp("2020-09-30")].copy()
    cfg = GarchConfig(horizon=5)

    fitted = fit_garch(train_df, cfg)
    out = make_garch_features(sample_garch_data, fitted, cfg)

    col = f"garch_sigma_fwd_h{cfg.horizon}"
    ticker_a = out[out["ticker"] == "A"].sort_values("date")
    last_40 = ticker_a.tail(40)[col]

    assert col in out.columns
    assert "garch_sigma_t" in out.columns
    assert "garch_var_fwd_h5" in out.columns
    assert "garch_resid" in out.columns
    assert "garch_z" in out.columns
    assert last_40.notna().sum() > 20


def test_garch_no_future_leakage_alignment(sample_garch_data):
    horizon = 5
    cfg = GarchConfig(horizon=horizon)

    cutoff = pd.Timestamp("2020-08-31")
    train_df = sample_garch_data[sample_garch_data["date"] <= cutoff].copy()
    fitted = fit_garch(train_df, cfg)

    base = sample_garch_data.sort_values(["ticker", "date"]).copy()
    altered = base.copy()

    # Perturb only far-future rows. Features at or before cutoff should remain unchanged.
    future_mask = altered["date"] > pd.Timestamp("2020-10-15")
    altered.loc[future_mask, "logret"] = altered.loc[future_mask, "logret"] * 15.0

    out_base = make_garch_features(base, fitted, cfg)
    out_alt = make_garch_features(altered, fitted, cfg)

    fwd_col = f"garch_sigma_fwd_h{horizon}"
    pre_cutoff = out_base["date"] <= cutoff
    diff = (out_base.loc[pre_cutoff, fwd_col] - out_alt.loc[pre_cutoff, fwd_col]).abs()
    assert diff.fillna(0.0).max() < 1e-12

    # Alignment sanity check: feature is not a trivial copy of contemporaneous returns.
    ticker_a = out_base[out_base["ticker"] == "A"].sort_values("date").copy()
    corr_now = ticker_a[[fwd_col, "logret"]].dropna().corr().iloc[0, 1]
    assert np.isfinite(corr_now)
    assert abs(corr_now) < 0.95

    realized_proxy = ticker_a["logret"].abs().shift(-horizon)
    corr_fwd = pd.concat([ticker_a[fwd_col], realized_proxy], axis=1).dropna().corr().iloc[0, 1]
    assert np.isfinite(corr_fwd)
