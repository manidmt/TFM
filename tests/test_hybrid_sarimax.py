'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-10

@description: Tests for hybrid SARIMAX + tabular volatility-regime classifier.
'''

import numpy as np
import pandas as pd
import pytest
import warnings

from quant_risk.models.econometric.sarimax import (
    SarimaxConfig,
    fit_sarimax,
    make_sarimax_features,
)


@pytest.fixture
def sample_train_data():
    """Create synthetic training data for SARIMAX."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    
    data = []
    for ticker in ["A", "B"]:
        vol = np.abs(np.random.randn(100) * 0.02 + 0.05)  # positive volatility
        data.append(pd.DataFrame({
            "ticker": ticker,
            "date": dates,
            "vol_fwd": vol,
            "rv_5": vol * np.random.uniform(0.8, 1.2, 100),
            "rv_20": vol * np.random.uniform(0.7, 1.3, 100),
        }))
    
    return pd.concat(data, ignore_index=True)


def test_sarimax_config_defaults():
    """Test SARIMAX config has sensible defaults."""
    cfg = SarimaxConfig()
    assert cfg.order == (1, 0, 1)
    assert cfg.seasonal_order == (0, 0, 0, 0)
    assert cfg.horizon == 5
    assert cfg.target_col == "rv_20"
    assert cfg.log_transform is True


def test_sarimax_no_future_leakage_alignment(sample_train_data):
    """Perturbing far-future rows must not change forecasts on earlier dates."""
    horizon = 5
    cutoff = pd.Timestamp("2020-03-20")

    base = sample_train_data.sort_values(["ticker", "date"]).copy()
    train_df = base[base["date"] <= cutoff].copy()

    cfg = SarimaxConfig(order=(1, 0, 0), horizon=horizon, target_col="rv_20", exog_cols=())
    models = fit_sarimax(train_df, cfg)

    altered = base.copy()
    future_mask = altered["date"] > pd.Timestamp("2020-04-20")
    altered.loc[future_mask, "rv_20"] = altered.loc[future_mask, "rv_20"] * 25.0

    out_base = make_sarimax_features(base, models, cfg)
    out_alt = make_sarimax_features(altered, models, cfg)

    fwd_col = f"sarimax_fcst_mean_h{horizon}"
    pre_cutoff = out_base["date"] <= cutoff
    diff = (out_base.loc[pre_cutoff, fwd_col] - out_alt.loc[pre_cutoff, fwd_col]).abs()
    assert diff.fillna(0.0).max() < 1e-12


def test_sarimax_alignment_sanity(sample_train_data):
    """Forecast should be more related to a future realized proxy than current return noise."""
    horizon = 5
    cfg = SarimaxConfig(order=(1, 0, 0), horizon=horizon, target_col="rv_20", exog_cols=())

    models = fit_sarimax(sample_train_data, cfg)
    out = make_sarimax_features(sample_train_data, models, cfg)

    fwd_col = f"sarimax_fcst_mean_h{horizon}"
    ticker_a = out[out["ticker"] == "A"].sort_values("date").copy()

    contemporaneous_proxy = ticker_a["vol_fwd"]
    future_proxy = ticker_a["vol_fwd"].shift(-horizon)

    pair_now = pd.concat([ticker_a[fwd_col], contemporaneous_proxy], axis=1).dropna()
    pair_fwd = pd.concat([ticker_a[fwd_col], future_proxy], axis=1).dropna()

    corr_now = pair_now.corr().iloc[0, 1]
    corr_fwd = pair_fwd.corr().iloc[0, 1]

    assert np.isfinite(corr_now)
    assert np.isfinite(corr_fwd)
    assert corr_fwd >= corr_now


def test_fit_sarimax_basic(sample_train_data):
    """Test SARIMAX fitting on multiple tickers."""
    cfg = SarimaxConfig(order=(1, 0, 0), horizon=5, exog_cols=())
    
    models = fit_sarimax(sample_train_data, cfg)
    
    assert "A" in models
    assert "B" in models
    assert models["A"] is not None
    assert models["B"] is not None


def test_fit_sarimax_with_exog(sample_train_data):
    """Test SARIMAX fitting with exogenous variables."""
    cfg = SarimaxConfig(order=(1, 0, 0), exog_cols=("rv_5", "rv_20"))
    
    models = fit_sarimax(sample_train_data, cfg)
    
    assert models["A"] is not None
    assert models["B"] is not None


def test_make_sarimax_features_columns(sample_train_data):
    """Test that SARIMAX features have expected columns."""
    cfg = SarimaxConfig(order=(1, 0, 0), horizon=5, exog_cols=())
    
    models = fit_sarimax(sample_train_data, cfg)
    features = make_sarimax_features(sample_train_data, models, cfg)
    
    expected_cols = [
        "sarimax_fcst_mean_h5",
        "sarimax_fcst_se_h5",
        "sarimax_ci_width_h5",
        "sarimax_resid",
        "sarimax_resid_std",
    ]
    
    for col in expected_cols:
        assert col in features.columns, f"Missing column: {col}"


def test_make_sarimax_features_no_leakage(sample_train_data):
    """Test that early rows have NaN for SARIMAX features (not enough history)."""
    cfg = SarimaxConfig(order=(1, 0, 0), horizon=5, exog_cols=())
    
    models = fit_sarimax(sample_train_data, cfg)
    features = make_sarimax_features(sample_train_data, models, cfg)
    
    # First few rows should be NaN (insufficient history)
    ticker_a = features[features["ticker"] == "A"].sort_values("date")
    first_10 = ticker_a.head(10)["sarimax_fcst_mean_h5"]
    
    assert first_10.isna().sum() > 0, "Expected NaNs in first rows due to insufficient history"


def test_make_sarimax_features_later_rows_have_values(sample_train_data):
    """Test that later rows have non-NaN SARIMAX forecasts."""
    cfg = SarimaxConfig(order=(1, 0, 0), horizon=5, exog_cols=())
    
    models = fit_sarimax(sample_train_data, cfg)
    features = make_sarimax_features(sample_train_data, models, cfg)
    
    # Later rows should have values
    ticker_a = features[features["ticker"] == "A"].sort_values("date")
    last_20 = ticker_a.tail(20)["sarimax_fcst_mean_h5"]
    
    assert last_20.notna().sum() > 10, "Expected non-NaN forecasts in later rows"


def test_make_sarimax_features_positive_forecasts(sample_train_data):
    """Test that volatility forecasts are positive (after inverse log)."""
    cfg = SarimaxConfig(order=(1, 0, 0), horizon=5, log_transform=True, exog_cols=())
    
    models = fit_sarimax(sample_train_data, cfg)
    features = make_sarimax_features(sample_train_data, models, cfg)
    
    valid_fcst = features["sarimax_fcst_mean_h5"].dropna()
    
    assert (valid_fcst >= 0).all(), "Volatility forecasts should be non-negative"


def test_sarimax_failed_fit_handling(sample_train_data):
    """Test that make_sarimax_features handles failed fits gracefully."""
    
    # Create invalid config that will likely fail
    cfg = SarimaxConfig(
        order=(10, 5, 10),  # overly complex
        horizon=5,
        exog_cols=()
    )
    
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Too few observations to estimate starting parameters*",
            category=UserWarning,
            module="statsmodels.tsa.statespace.sarimax",
        )
        models = fit_sarimax(sample_train_data.head(20), cfg)
    
    # Should still return features with NaNs for failed tickers
    features = make_sarimax_features(sample_train_data, models, cfg)
    
    assert "sarimax_fcst_mean_h5" in features.columns
    assert len(features) == len(sample_train_data)


def test_sarimax_features_preserve_original_columns(sample_train_data):
    """Test that make_sarimax_features preserves original data columns."""
    cfg = SarimaxConfig(order=(1, 0, 0), horizon=5, exog_cols=())
    
    models = fit_sarimax(sample_train_data, cfg)
    features = make_sarimax_features(sample_train_data, models, cfg)
    
    # Original columns should still be present
    assert "ticker" in features.columns
    assert "date" in features.columns
    assert "vol_fwd" in features.columns
    assert len(features) == len(sample_train_data)


def test_sarimax_features_same_row_count_per_ticker(sample_train_data):
    """Test that each ticker has the same number of rows before and after."""
    cfg = SarimaxConfig(order=(1, 0, 0), horizon=5, exog_cols=())
    
    models = fit_sarimax(sample_train_data, cfg)
    features = make_sarimax_features(sample_train_data, models, cfg)
    
    for ticker in sample_train_data["ticker"].unique():
        orig_count = len(sample_train_data[sample_train_data["ticker"] == ticker])
        feat_count = len(features[features["ticker"] == ticker])
        assert orig_count == feat_count, f"Row count mismatch for {ticker}"
