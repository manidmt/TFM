'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-12

@description: Smoke tests for GKG change/no-change detector module.
'''

from __future__ import annotations

import numpy as np

from quant_risk.models.tabular.gkg_change_detector import (
    GkgChangeDetectorConfig,
    build_design_matrix,
    fit,
    predict_proba,
    select_gkg_feature_columns,
)


def test_select_gkg_feature_columns_filters_news_only():
    cols = [
        "attn_z_macro_us",
        "tone_change_1d_macro_us",
        "topic_share_z_rates_yields",
        "news_count_mean_w10_crypto_market",
        "rv_20",
        "sigma_t",
        "attn_z_x_rv20_macro_us",
        "garch_sigma_t",
    ]
    selected = select_gkg_feature_columns(cols)
    assert "attn_z_macro_us" in selected
    assert "tone_change_1d_macro_us" in selected
    assert "topic_share_z_rates_yields" in selected
    assert "news_count_mean_w10_crypto_market" in selected
    assert "rv_20" not in selected
    assert "sigma_t" not in selected
    assert "attn_z_x_rv20_macro_us" not in selected


def test_build_design_matrix_with_context():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(12, 5)).astype(np.float32)
    cols = [
        "attn_z_macro_us",
        "tone_change_1d_macro_us",
        "rv_20",
        "sigma_t",
        "attn_z_x_rv20_macro_us",
    ]
    regime_prev = rng.integers(0, 3, size=12)
    sigma_t = x[:, 3]
    xm, names = build_design_matrix(
        X_all=x,
        feature_cols=cols,
        context_cols=("regime_prev", "sigma_t"),
        regime_prev=regime_prev,
        sigma_t=sigma_t,
    )
    assert xm.shape[0] == 12
    assert xm.shape[1] >= 3
    assert "ctx_regime_prev" in names
    assert "ctx_sigma_t" in names


def test_fit_predict_logit_outputs_calibrated_probabilities():
    rng = np.random.default_rng(123)
    n = 80
    cols = [
        "attn_z_macro_us",
        "tone_change_1d_macro_us",
        "topic_share_z_macro_us",
        "rv_20",
        "sigma_t",
    ]
    x = rng.normal(size=(n, len(cols))).astype(np.float32)
    regime_prev = rng.integers(0, 3, size=n)
    sigma_t = np.abs(x[:, 4]) + 0.1
    x_dm, names = build_design_matrix(
        X_all=x,
        feature_cols=cols,
        context_cols=("regime_prev", "sigma_t"),
        regime_prev=regime_prev,
        sigma_t=sigma_t,
    )

    # Balanced-ish binary target with signal in first feature.
    y = (x_dm[:, 0] + 0.2 * x_dm[:, 1] > 0.0).astype(int)
    y[:10] = 0
    y[10:20] = 1

    cfg = GkgChangeDetectorConfig(model_type="logit", calibration_method="platt")
    artifacts = fit(cfg, x_dm, y, feature_names=names)
    proba = predict_proba(artifacts, x_dm)

    assert proba.shape == (n, 2)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(n), atol=1e-6)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
