'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-12

@description: Tests for hybrid GARCH + tabular volatility-regime classifier.
'''

import numpy as np
import pandas as pd
import pytest

from quant_risk.models.econometric.garch import (
    GarchConfig,
    fit_garch,
    make_garch_features,
)


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
def sample_train_data() -> pd.DataFrame:
    """Create synthetic training data for GARCH."""
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


def test_garch_config_defaults():
    """Test GARCH config has sensible defaults."""
    cfg = GarchConfig()
    assert cfg.p == 1
    assert cfg.q == 1
    assert cfg.horizon == 5
    assert cfg.target_col == "logret"
    assert cfg.dist == "normal"
    assert cfg.mean == "zero"
    assert cfg.vol == "Garch"
    assert cfg.annualize is False
    assert cfg.scale == 100.0


def test_fit_garch_returns_models_per_ticker(sample_train_data):
    """Test GARCH fitting on multiple tickers."""
    train_df = sample_train_data[sample_train_data["date"] <= pd.Timestamp("2020-09-30")].copy()
    cfg = GarchConfig(horizon=5)

    fitted = fit_garch(train_df, cfg)

    assert "A" in fitted
    assert "B" in fitted
    assert fitted["A"] is not None
    assert fitted["B"] is not None
    assert "params" in fitted["A"]
    assert "params" in fitted["B"]


def test_make_garch_features_columns(sample_train_data):
    """Test that GARCH features have expected columns."""
    train_df = sample_train_data[sample_train_data["date"] <= pd.Timestamp("2020-09-30")].copy()
    cfg = GarchConfig(horizon=5)

    fitted = fit_garch(train_df, cfg)
    features = make_garch_features(sample_train_data, fitted, cfg)

    expected_cols = [
        "garch_sigma_t",
        "garch_sigma_fwd_h5",
        "garch_var_fwd_h5",
        "garch_resid",
        "garch_z",
    ]

    for col in expected_cols:
        assert col in features.columns, f"Missing column: {col}"


def test_make_garch_features_no_leakage(sample_train_data):
    """Test that early rows have NaN for GARCH features (not enough history)."""
    train_df = sample_train_data[sample_train_data["date"] <= pd.Timestamp("2020-09-30")].copy()
    cfg = GarchConfig(horizon=5)

    fitted = fit_garch(train_df, cfg)
    features = make_garch_features(sample_train_data, fitted, cfg)

    ticker_a = features[features["ticker"] == "A"].sort_values("date")
    first_20 = ticker_a.head(20)["garch_sigma_fwd_h5"]

    assert first_20.isna().sum() > 0, "Expected NaNs in first rows due to insufficient history"


def test_make_garch_features_later_rows_have_values(sample_train_data):
    """Test that later rows have non-NaN GARCH forecasts."""
    train_df = sample_train_data[sample_train_data["date"] <= pd.Timestamp("2020-09-30")].copy()
    cfg = GarchConfig(horizon=5)

    fitted = fit_garch(train_df, cfg)
    features = make_garch_features(sample_train_data, fitted, cfg)

    ticker_a = features[features["ticker"] == "A"].sort_values("date")
    last_40 = ticker_a.tail(40)["garch_sigma_fwd_h5"]

    assert last_40.notna().sum() > 20, "Expected non-NaN forecasts in later rows"


def test_make_garch_features_positive_forecasts(sample_train_data):
    """Test that GARCH volatility forecasts are non-negative."""
    train_df = sample_train_data[sample_train_data["date"] <= pd.Timestamp("2020-09-30")].copy()
    cfg = GarchConfig(horizon=5)

    fitted = fit_garch(train_df, cfg)
    features = make_garch_features(sample_train_data, fitted, cfg)

    valid_fcst = features["garch_sigma_fwd_h5"].dropna()
    assert (valid_fcst >= 0).all(), "Volatility forecasts should be non-negative"


def test_garch_no_future_leakage_alignment(sample_train_data):
    """Test alignment and no look-ahead leakage in GARCH features."""
    horizon = 5
    cutoff = pd.Timestamp("2020-08-31")

    train_df = sample_train_data[sample_train_data["date"] <= cutoff].copy()
    cfg = GarchConfig(horizon=horizon)
    fitted = fit_garch(train_df, cfg)

    base = sample_train_data.sort_values(["ticker", "date"]).copy()
    altered = base.copy()

    future_mask = altered["date"] > pd.Timestamp("2020-10-15")
    altered.loc[future_mask, "logret"] = altered.loc[future_mask, "logret"] * 15.0

    out_base = make_garch_features(base, fitted, cfg)
    out_alt = make_garch_features(altered, fitted, cfg)

    fwd_col = f"garch_sigma_fwd_h{horizon}"
    pre_cutoff = out_base["date"] <= cutoff
    diff = (out_base.loc[pre_cutoff, fwd_col] - out_alt.loc[pre_cutoff, fwd_col]).abs()
    assert diff.fillna(0.0).max() < 1e-12

    ticker_a = out_base[out_base["ticker"] == "A"].sort_values("date").copy()
    contemporaneous_proxy = ticker_a["logret"].abs()
    corr_now = pd.concat([ticker_a[fwd_col], contemporaneous_proxy], axis=1).dropna().corr().iloc[0, 1]
    assert np.isfinite(corr_now)

    realized_proxy = ticker_a["logret"].abs().shift(-horizon)
    corr_fwd = pd.concat([ticker_a[fwd_col], realized_proxy], axis=1).dropna().corr().iloc[0, 1]
    assert np.isfinite(corr_fwd)
    assert corr_fwd > -0.05
    assert abs(corr_now) < 0.95


def test_garch_failed_fit_handling(sample_train_data):
    """Test that make_garch_features handles failed fits gracefully."""
    train_df = sample_train_data[sample_train_data["date"] <= pd.Timestamp("2020-03-15")].copy()

    cfg = GarchConfig(p=20, q=20, horizon=5)
    fitted = fit_garch(train_df, cfg)
    features = make_garch_features(sample_train_data, fitted, cfg)

    assert "garch_sigma_fwd_h5" in features.columns
    assert len(features) == len(sample_train_data)


def test_garch_features_preserve_original_columns(sample_train_data):
    """Test that make_garch_features preserves original data columns."""
    train_df = sample_train_data[sample_train_data["date"] <= pd.Timestamp("2020-09-30")].copy()
    cfg = GarchConfig(horizon=5)

    fitted = fit_garch(train_df, cfg)
    features = make_garch_features(sample_train_data, fitted, cfg)

    assert "ticker" in features.columns
    assert "date" in features.columns
    assert "logret" in features.columns
    assert len(features) == len(sample_train_data)


def test_garch_features_same_row_count_per_ticker(sample_train_data):
    """Test that each ticker has the same number of rows before and after."""
    train_df = sample_train_data[sample_train_data["date"] <= pd.Timestamp("2020-09-30")].copy()
    cfg = GarchConfig(horizon=5)

    fitted = fit_garch(train_df, cfg)
    features = make_garch_features(sample_train_data, fitted, cfg)

    for ticker in sample_train_data["ticker"].unique():
        orig_count = len(sample_train_data[sample_train_data["ticker"] == ticker])
        feat_count = len(features[features["ticker"] == ticker])
        assert orig_count == feat_count, f"Row count mismatch for {ticker}"
