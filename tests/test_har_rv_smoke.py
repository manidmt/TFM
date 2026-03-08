'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-07

@description: Smoke tests for HAR-RV econometric feature generator.
'''

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_risk.models.econometric.har_rv import HarRvConfig, fit_har_rv, make_har_rv_features


def _sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=260)
    rows = []
    for ticker in ["A", "B"]:
        logret = rng.normal(0.0, 0.01, size=len(dates))
        rv20 = pd.Series(logret).rolling(20).std().bfill().to_numpy()
        macro = rng.normal(0.0, 1.0, size=len(dates)).cumsum()
        rows.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "logret": logret,
                    "rv_20": rv20,
                    "net_liquidity_diff": np.concatenate([[0.0], np.diff(macro)]),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def test_har_rv_smoke_features_non_empty():
    full_df = _sample_df()
    train_df = full_df[full_df["date"] <= pd.Timestamp("2020-10-30")].copy()

    cfg = HarRvConfig(
        horizon=5,
        target_col="rv_20",
        lag_1=1,
        lag_week=5,
        lag_month=22,
        exog_cols=("net_liquidity_diff",),
    )
    fitted = fit_har_rv(train_df, cfg)
    out = make_har_rv_features(full_df, fitted, cfg)

    assert "A" in fitted and "B" in fitted
    assert "har_fcst_mean_h5" in out.columns
    assert "har_resid" in out.columns
    assert out["har_fcst_mean_h5"].notna().sum() > 0
    assert out["har_resid"].notna().sum() > 0
