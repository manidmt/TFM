'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-07

@description: Smoke tests for skew-t support in GARCH family models.
'''

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_risk.models.econometric.egarch import EgarchConfig, fit_egarch, make_egarch_features
from quant_risk.models.econometric.garch import GarchConfig, fit_garch, make_garch_features
from quant_risk.models.econometric.gjrgarch import (
    GjrGarchConfig,
    fit_gjrgarch,
    make_gjrgarch_features,
)


def _simulate_returns(n: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    sigma2 = np.full(n, 1e-4, dtype=float)
    ret = np.zeros(n, dtype=float)

    omega = 1e-6
    alpha = 0.08
    beta = 0.90
    gamma = 0.05

    for t in range(1, n):
        shock = ret[t - 1] ** 2
        asym = shock if ret[t - 1] < 0 else 0.0
        sigma2[t] = omega + alpha * shock + gamma * asym + beta * sigma2[t - 1]
        ret[t] = np.sqrt(max(sigma2[t], 1e-12)) * eps[t]

    return ret


def _sample_df() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=220, freq="B")
    rows = []
    for i, ticker in enumerate(["A", "B"]):
        rows.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "logret": _simulate_returns(len(dates), seed=300 + i),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_garch_skewt_smoke_features_non_empty():
    full_df = _sample_df()
    train_df = full_df[full_df["date"] <= pd.Timestamp("2021-08-31")].copy()

    cfg = GarchConfig(horizon=5, dist="skewt")
    fitted = fit_garch(train_df, cfg)
    out = make_garch_features(full_df, fitted, cfg)

    assert "A" in fitted and "B" in fitted
    assert "garch_sigma_fwd_h5" in out.columns
    assert out["garch_sigma_fwd_h5"].notna().sum() > 0
    assert (out["garch_sigma_fwd_h5"].dropna() >= 0).all()


def test_egarch_skewt_smoke_features_non_empty():
    full_df = _sample_df()
    train_df = full_df[full_df["date"] <= pd.Timestamp("2021-08-31")].copy()

    cfg = EgarchConfig(horizon=5, dist="skewt")
    fitted = fit_egarch(train_df, cfg)
    out = make_egarch_features(full_df, fitted, cfg)

    assert "A" in fitted and "B" in fitted
    assert "egarch_sigma_fwd_h5" in out.columns
    assert out["egarch_sigma_fwd_h5"].notna().sum() > 0
    assert (out["egarch_sigma_fwd_h5"].dropna() >= 0).all()


def test_gjrgarch_skewt_smoke_features_non_empty():
    full_df = _sample_df()
    train_df = full_df[full_df["date"] <= pd.Timestamp("2021-08-31")].copy()

    cfg = GjrGarchConfig(horizon=5, o=1, dist="skewt")
    fitted = fit_gjrgarch(train_df, cfg)
    out = make_gjrgarch_features(full_df, fitted, cfg)

    assert "A" in fitted and "B" in fitted
    assert "gjrgarch_sigma_fwd_h5" in out.columns
    assert out["gjrgarch_sigma_fwd_h5"].notna().sum() > 0
    assert (out["gjrgarch_sigma_fwd_h5"].dropna() >= 0).all()

