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


def test_gkg_change_gate_routes_low_pchange_to_persistence():
    mod = _load_walk_forward_module()
    proba = np.array(
        [
            [0.05, 0.05, 0.90],
            [0.80, 0.10, 0.10],
        ],
        dtype=float,
    )
    y_persist = np.array([1, 2], dtype=int)
    p_change = np.array([0.15, 0.90], dtype=float)
    out = mod._blend_strategy_predict(
        proba_chain=proba,
        y_persist=y_persist,
        alpha=1.0,
        beta=0.0,
        class_thresholds=np.array([1.0, 1.0, 1.0], dtype=float),
        gate_threshold=0.0,
        p_change=p_change,
        change_gate_threshold=0.50,
    )
    pred = np.asarray(out["pred"], dtype=int)
    assert pred.tolist() == [1, 0]
    assert np.asarray(out["conf_gate_mask"], dtype=bool).tolist() == [False, False]
    assert np.asarray(out["change_gate_mask"], dtype=bool).tolist() == [True, False]
    assert np.asarray(out["gate_mask"], dtype=bool).tolist() == [True, False]


def test_gkg_change_alpha_weight_attenuates_alpha_toward_persistence():
    mod = _load_walk_forward_module()
    proba = np.array([[0.60, 0.39, 0.01]], dtype=float)
    y_persist = np.array([1], dtype=int)
    out = mod._blend_strategy_predict(
        proba_chain=proba,
        y_persist=y_persist,
        alpha=1.0,
        beta=0.0,
        class_thresholds=np.array([1.0, 1.0, 1.0], dtype=float),
        gate_threshold=0.0,
        p_change=np.array([0.20], dtype=float),
        change_gate_threshold=0.0,
        change_alpha_weight=1.0,
    )
    pred = np.asarray(out["pred"], dtype=int)
    alpha_vec = np.asarray(out["alpha_vec"], dtype=float)
    alpha_vec_base = np.asarray(out["alpha_vec_base"], dtype=float)
    alpha_multiplier = np.asarray(out["change_alpha_multiplier"], dtype=float)
    assert pred.tolist() == [1]
    np.testing.assert_allclose(alpha_vec_base, np.array([1.0], dtype=float))
    np.testing.assert_allclose(alpha_multiplier, np.array([0.20], dtype=float))
    np.testing.assert_allclose(alpha_vec, np.array([0.20], dtype=float))
    assert np.asarray(out["gate_mask"], dtype=bool).tolist() == [False]


def test_gkg_change_alpha_weights_by_asset_override_resolution():
    mod = _load_walk_forward_module()
    overrides = mod._parse_gkg_change_alpha_weights_by_asset(
        "BTC-USD=0.0,0.1,0.2;TLT=0.0,0.05,0.1"
    )
    assert overrides["BTC-USD"] == (0.0, 0.1, 0.2)
    assert overrides["TLT"] == (0.0, 0.05, 0.1)
    grid_btc = mod._resolve_asset_specific_grid(
        default_grid=(0.0, 0.25, 0.5),
        asset_overrides=overrides,
        asset_name="BTC-USD",
        tickers=("BTC-USD",),
    )
    grid_tlt = mod._resolve_asset_specific_grid(
        default_grid=(0.0, 0.25, 0.5),
        asset_overrides=overrides,
        asset_name="TLT",
        tickers=("TLT",),
    )
    grid_default = mod._resolve_asset_specific_grid(
        default_grid=(0.0, 0.25, 0.5),
        asset_overrides=overrides,
        asset_name="^GSPC",
        tickers=("^GSPC",),
    )
    assert grid_btc == (0.0, 0.1, 0.2)
    assert grid_tlt == (0.0, 0.05, 0.1)
    assert grid_default == (0.0, 0.25, 0.5)


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
