'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-08

@description: Smoke tests for persistence-aware gating and calibration utilities.
'''

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_walk_forward_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "walk_forward_chain_tab.py"
    spec = importlib.util.spec_from_file_location("walk_forward_chain_tab", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gating_routes_low_confidence_to_persistence():
    mod = _load_walk_forward_module()
    proba = np.array(
        [
            [0.34, 0.33, 0.33],
            [0.90, 0.05, 0.05],
        ],
        dtype=float,
    )
    y_persist = np.array([2, 1], dtype=int)
    out = mod._blend_strategy_predict(
        proba_chain=proba,
        y_persist=y_persist,
        alpha=1.0,
        beta=0.0,
        class_thresholds=np.array([1.0, 1.0, 1.0], dtype=float),
        gate_threshold=0.60,
    )
    pred = np.asarray(out["pred"], dtype=int)
    assert pred.tolist() == [2, 0]
    assert np.asarray(out["gate_mask"], dtype=bool).tolist() == [True, False]


def test_class_thresholds_change_argmax_decision():
    mod = _load_walk_forward_module()
    proba = np.array([[0.40, 0.35, 0.25]], dtype=float)
    pred_default = mod._predict_with_class_thresholds(proba)
    pred_shifted = mod._predict_with_class_thresholds(
        proba,
        class_thresholds=np.array([1.20, 0.80, 1.00], dtype=float),
    )
    assert int(pred_default[0]) == 0
    assert int(pred_shifted[0]) == 1


def test_platt_calibration_preserves_probability_shape():
    mod = _load_walk_forward_module()
    proba_train = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.20, 0.70, 0.10],
            [0.10, 0.20, 0.70],
            [0.75, 0.20, 0.05],
            [0.05, 0.80, 0.15],
            [0.10, 0.15, 0.75],
        ],
        dtype=float,
    )
    y_train = np.array([0, 1, 2, 0, 1, 2], dtype=int)
    calibrator = mod._fit_probability_calibrator(proba_train, y_train, method="platt")

    proba_eval = np.array(
        [
            [0.55, 0.30, 0.15],
            [0.10, 0.50, 0.40],
        ],
        dtype=float,
    )
    proba_cal = mod._apply_probability_calibrator(proba_eval, calibrator)
    assert proba_cal.shape == proba_eval.shape
    assert np.isfinite(proba_cal).all()
    np.testing.assert_allclose(np.sum(proba_cal, axis=1), np.ones(proba_cal.shape[0]), atol=1e-6)
