"""Smoke tests for the MLflow helper functions added to walk_forward_chain_tab.py.

These tests verify that the param/metric/tag builder functions work correctly in
isolation, without requiring a real MLflow tracking server or any data files.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_walk_forward_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "walk_forward_chain_tab.py"
    spec = importlib.util.spec_from_file_location("walk_forward_chain_tab", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_args(**overrides):
    """Return a minimal Namespace with all required fields for the builder helpers."""
    defaults = dict(
        horizon=5,
        tabular_model="xgb",
        per_ticker=False,
        grid_profile="promising",
        min_train_end="2018-12-31",
        max_valid_end="2023-12-31",
        valid_months=12,
        step_months=6,
        max_struct_configs=0,
        max_model_configs=0,
        max_xgb_configs=0,
        min_folds_ok=2,
        min_positive_rate=0.5,
        min_non_transition_delta=-0.02,
        min_high_vol_recall_delta=0.0,
        stability_lambda=1.0,
        positive_rate_lambda=0.5,
        non_transition_penalty_lambda=1.0,
        transition_bonus_lambda=0.0,
        prune_after_folds=2,
        prune_delta_threshold=-0.03,
        use_blend=False,
        blend_alphas="0.0,1.0",
        blend_conf_betas="0.0",
        class_threshold_grid="1,1,1",
        gate_thresholds="0.0",
        calibration_method="none",
        use_gkg_change_detector=False,
        use_gkg_change_gate=False,
        use_gkg_change_alpha=False,
        gkg_change_model="logit",
        gkg_change_context_cols="regime_prev",
        gkg_change_calibration_method="none",
        gkg_change_gate_thresholds="0.0",
        gkg_change_alpha_weights="0.0",
        gkg_change_alpha_weights_by_asset="",
        gkg_change_blend_hook_weight=0.0,
        outdir="runs/test",
        use_mlflow=False,
        mlflow_tracking_uri="file:./mlruns",
        mlflow_experiment="walk_forward_chain_tab",
        mlflow_parent_run_name="",
        mlflow_strict=False,
        mlflow_tags_json="",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_sources_cfg():
    return {
        "db": {"path": "data/db/financial_data.duckdb"},
        "gkg": {
            "profile_name": "gkg_v1_light",
            "profile_version": "2026-03-13",
            "profile_mode": "light_daily",
            "enabled": True,
            "start": "2020-01-01",
            "store_raw": False,
            "sample_interval_minutes": 180,
            "max_files_per_day": 8,
            "publication_lag_bdays": 1,
        },
    }


def _make_features_cfg():
    return {
        "split": {"train_end": "2020-12-31", "val_end": "2023-12-31"},
        "news_features": {
            "profile_name": "gkg_v1_features_compact",
            "profile_version": "2026-03-13",
            "source_table": "news_features_daily",
            "windows": [3, 10, 20],
            "compact_mode": True,
            "include_interactions": True,
        },
    }


def test_build_mlflow_parent_tags_has_required_keys():
    mod = _load_walk_forward_module()
    args = _make_args()
    tags = mod._build_mlflow_parent_tags(args, "abc1234", _make_sources_cfg())
    for key in ("script", "component", "git_sha", "horizon", "tabular_model", "news_profile"):
        assert key in tags, f"Missing tag: {key}"
    assert tags["git_sha"] == "abc1234"
    assert tags["horizon"] == "5"
    assert tags["news_profile"] == "gkg_v1_light"


def test_build_mlflow_child_tags_has_required_keys():
    mod = _load_walk_forward_module()
    args = _make_args()
    tags = mod._build_mlflow_child_tags("GSPC", ("^GSPC",), args)
    assert tags["asset"] == "GSPC"
    assert tags["tickers"] == "^GSPC"
    assert tags["horizon"] == "5"
    assert tags["tabular_model"] == "xgb"


def test_build_mlflow_child_params_has_required_keys():
    mod = _load_walk_forward_module()
    args = _make_args()
    params = mod._build_mlflow_child_params(args, _make_sources_cfg(), _make_features_cfg())
    required_keys = [
        "horizon", "tabular_model", "grid_profile",
        "use_gkg_change_detector", "gkg_profile_name",
        "news_feature_profile_name", "news_windows",
    ]
    for k in required_keys:
        assert k in params, f"Missing param: {k}"
    assert params["gkg_profile_name"] == "gkg_v1_light"
    assert params["news_feature_profile_name"] == "gkg_v1_features_compact"


def test_build_mlflow_child_metrics_from_best_and_final_cmp():
    mod = _load_walk_forward_module()
    best = {
        "robust_score": 0.08,
        "n_folds_ok": 3,
        "mean_delta_macro_vs_persistence": 0.05,
        "mean_delta_high_vol_recall_vs_persistence": 0.03,
        "mean_delta_transition_macro_f1_vs_persistence": 0.02,
        "mean_delta_non_transition_macro_f1_vs_persistence": -0.01,
        "oof_delta_macro_vs_persistence": 0.06,
        "struct_id": 0,
        "xgb_id": 2,
        "oof_selected_alpha": 0.8,
        "oof_selected_beta": 0.0,
        "oof_selected_gate_threshold": 0.6,
        "oof_selected_change_gate_threshold": 0.5,
        "oof_selected_change_alpha_weight": 0.25,
    }
    final_cmp = {
        "delta_test_macro_f1_vs_persistence": 0.07,
        "delta_test_high_vol_recall_vs_persistence": 0.04,
    }
    metrics = mod._build_mlflow_child_metrics(best, final_cmp)
    assert metrics["robust_score"] == 0.08
    assert metrics["n_folds_ok"] == 3
    assert metrics["delta_test_macro_f1_vs_persistence"] == 0.07
    assert metrics["selected_struct_id"] == 0
    assert metrics["selected_blend_alpha"] == 0.8


def test_mlflow_cfg_disabled_does_not_call_mlflow():
    """When use_mlflow=False, tracking helpers must be pure no-ops."""
    from quant_risk.tracking.mlflow_utils import MlflowConfig, start_parent_run
    from unittest.mock import patch
    cfg = MlflowConfig(enabled=False)
    with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
        result = start_parent_run(cfg, run_name="test", tags={})
        assert result is None
        mock_mlflow.start_run.assert_not_called()
