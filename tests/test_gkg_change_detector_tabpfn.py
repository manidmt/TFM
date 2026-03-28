'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Tests for TabPFN support in the GKG change detector.
'''
import sys
import unittest.mock as mock

import numpy as np
import pytest

from quant_risk.models.tabular.gkg_change_detector import (
    GkgChangeDetectorConfig,
    fit,
    make_model,
    predict_proba,
)


def test_tabpfn_config_defaults():
    cfg = GkgChangeDetectorConfig(model_type="tabpfn")
    assert cfg.tabpfn_n_estimators == 8
    assert cfg.tabpfn_softmax_temperature == pytest.approx(0.9)
    assert cfg.tabpfn_balance_probabilities is False


def test_make_model_tabpfn_raises_import_error():
    with mock.patch.dict(sys.modules, {"tabpfn": None}):
        with pytest.raises(ImportError, match="tabpfn no está instalado"):
            make_model(GkgChangeDetectorConfig(model_type="tabpfn"))


@pytest.mark.parametrize("cal_method", ["none", "platt", "isotonic"])
def test_fit_tabpfn_skips_calibration(cal_method):
    cfg = GkgChangeDetectorConfig(
        model_type="tabpfn",
        calibration_method=cal_method,
    )
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 5)).astype(np.float32)
    y = np.array([0, 1] * 15)
    feature_names = [f"f{i}" for i in range(5)]

    mock_clf = mock.MagicMock()
    mock_clf.predict_proba.return_value = np.column_stack(
        [np.full(30, 0.4), np.full(30, 0.6)]
    )

    with mock.patch("tabpfn.TabPFNClassifier", return_value=mock_clf):
        artifacts = fit(cfg, X, y, feature_names=feature_names)

    assert artifacts.calibrator["method"] == "none"
    assert artifacts.calibrator["model"] is None


def test_fit_tabpfn_produces_valid_probabilities():
    cfg = GkgChangeDetectorConfig(model_type="tabpfn", calibration_method="platt")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6)).astype(np.float32)
    y = np.array([0, 1] * 20)
    feature_names = [f"feat_{i}" for i in range(6)]

    n_rows = X.shape[0]
    raw_proba = np.column_stack(
        [rng.uniform(0.2, 0.5, n_rows), rng.uniform(0.5, 0.8, n_rows)]
    ).astype(np.float64)

    mock_clf = mock.MagicMock()
    mock_clf.predict_proba.return_value = raw_proba

    with mock.patch("tabpfn.TabPFNClassifier", return_value=mock_clf):
        artifacts = fit(cfg, X, y, feature_names=feature_names)
        proba = predict_proba(artifacts, X)

    # shape
    assert proba.shape == (n_rows, 2)
    # values in [0, 1]
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    # rows sum to ~1 — safe: predict_proba() reconstructs [1-p_change, p_change] from the
    # scalar in mock column 1; the mock's raw columns need not be normalised
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(n_rows), atol=1e-6)
    # calibrator unchanged (no calibration applied)
    assert artifacts.calibrator == {"method": "none", "model": None}
