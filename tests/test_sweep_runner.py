"""Unit tests for sweep_runner.py."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

# Add scripts/ to path so we can import sweep_runner
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import sweep_runner


# ---------------------------------------------------------------------------
# load_sweep_config
# ---------------------------------------------------------------------------

def test_load_sweep_config_returns_dict(tmp_path):
    cfg = {
        "sweep": {"n_trials": 5, "mlflow_experiment": "test_exp", "assets": ["^GSPC"]},
        "search_space": {
            "tabular_model": {"type": "categorical", "choices": ["xgb", "tabpfn"]},
        },
    }
    p = tmp_path / "sweep.yaml"
    p.write_text(yaml.dump(cfg))
    result = sweep_runner.load_sweep_config(str(p))
    assert result["sweep"]["n_trials"] == 5
    assert result["sweep"]["assets"] == ["^GSPC"]
    assert "search_space" in result


def test_load_sweep_config_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        sweep_runner.load_sweep_config("/nonexistent/path.yaml")


# ---------------------------------------------------------------------------
# build_trial_args
# ---------------------------------------------------------------------------

def _make_trial(params: dict):
    """Return a mock Optuna trial that returns params on suggest_* calls."""
    trial = MagicMock()
    trial.number = 3

    def suggest_categorical(name, choices):
        return params.get(name, choices[0])

    def suggest_int(name, low, high, **kw):
        return params.get(name, low)

    def suggest_float(name, low, high, **kw):
        return params.get(name, low)

    trial.suggest_categorical.side_effect = suggest_categorical
    trial.suggest_int.side_effect = suggest_int
    trial.suggest_float.side_effect = suggest_float
    return trial


def test_build_trial_args_tabpfn_passes_tabpfn_flags():
    search_space = {
        "tabular_model": {"type": "categorical", "choices": ["tabpfn"]},
        "tabpfn_n_estimators": {"type": "int", "low": 4, "high": 16},
        "tabpfn_softmax_temperature": {"type": "float", "low": 0.5, "high": 1.5},
        "xgb_max_depth": {"type": "int", "low": 3, "high": 8},
        "use_blend": {"type": "categorical", "choices": ["false"]},
        "use_gkg_change_detector": {"type": "categorical", "choices": ["false"]},
        "gkg_change_model": {"type": "categorical", "choices": ["logit"]},
        "gkg_tabpfn_n_estimators": {"type": "int", "low": 4, "high": 16},
        "chain_variants_config": {"type": "categorical", "choices": ["config/chain_variants.yaml"]},
    }
    params = {
        "tabular_model": "tabpfn",
        "tabpfn_n_estimators": 8,
        "tabpfn_softmax_temperature": 0.9,
        "use_blend": "false",
        "use_gkg_change_detector": "false",
        "gkg_change_model": "logit",
        "gkg_tabpfn_n_estimators": 6,
        "xgb_max_depth": 5,
        "chain_variants_config": "config/chain_variants.yaml",
    }
    trial = _make_trial(params)
    args = sweep_runner.build_trial_args(trial, search_space, "^GSPC", [])
    args_str = " ".join(args)
    assert "--tabular_model tabpfn" in args_str
    assert "--tabpfn_n_estimators 8" in args_str
    assert "--tabpfn_softmax_temperature" in args_str
    # XGB flags must NOT appear when tabular_model=tabpfn
    assert "--xgb_max_depth" not in args_str


def test_build_trial_args_xgb_passes_xgb_flags():
    search_space = {
        "tabular_model": {"type": "categorical", "choices": ["xgb"]},
        "tabpfn_n_estimators": {"type": "int", "low": 4, "high": 16},
        "tabpfn_softmax_temperature": {"type": "float", "low": 0.5, "high": 1.5},
        "xgb_max_depth": {"type": "int", "low": 3, "high": 8},
        "xgb_learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
        "use_blend": {"type": "categorical", "choices": ["true"]},
        "use_gkg_change_detector": {"type": "categorical", "choices": ["false"]},
        "gkg_change_model": {"type": "categorical", "choices": ["logit"]},
        "gkg_tabpfn_n_estimators": {"type": "int", "low": 4, "high": 16},
        "chain_variants_config": {"type": "categorical", "choices": ["config/chain_variants.yaml"]},
    }
    params = {
        "tabular_model": "xgb",
        "xgb_max_depth": 5,
        "xgb_learning_rate": 0.05,
        "use_blend": "true",
        "use_gkg_change_detector": "false",
        "gkg_change_model": "logit",
        "gkg_tabpfn_n_estimators": 4,
        "tabpfn_n_estimators": 8,
        "tabpfn_softmax_temperature": 0.9,
        "chain_variants_config": "config/chain_variants.yaml",
    }
    trial = _make_trial(params)
    args = sweep_runner.build_trial_args(trial, search_space, "^GSPC", [])
    args_str = " ".join(args)
    assert "--xgb_max_depth 5" in args_str
    assert "--xgb_learning_rate" in args_str
    assert "--use_blend" in args_str
    # TabPFN flags must NOT appear when tabular_model=xgb
    assert "--tabpfn_n_estimators" not in args_str


def test_build_trial_args_gkg_tabpfn_only_when_active():
    """gkg_tabpfn_n_estimators is only passed when use_gkg_change_detector=true AND gkg_change_model=tabpfn."""
    search_space = {
        "tabular_model": {"type": "categorical", "choices": ["xgb"]},
        "tabpfn_n_estimators": {"type": "int", "low": 4, "high": 16},
        "tabpfn_softmax_temperature": {"type": "float", "low": 0.5, "high": 1.5},
        "xgb_max_depth": {"type": "int", "low": 3, "high": 8},
        "xgb_learning_rate": {"type": "float", "low": 0.01, "high": 0.2},
        "use_blend": {"type": "categorical", "choices": ["false"]},
        "use_gkg_change_detector": {"type": "categorical", "choices": ["true"]},
        "gkg_change_model": {"type": "categorical", "choices": ["tabpfn"]},
        "gkg_tabpfn_n_estimators": {"type": "int", "low": 4, "high": 16},
        "chain_variants_config": {"type": "categorical", "choices": ["config/chain_variants.yaml"]},
    }
    params = {
        "tabular_model": "xgb",
        "use_gkg_change_detector": "true",
        "gkg_change_model": "tabpfn",
        "gkg_tabpfn_n_estimators": 12,
        "xgb_max_depth": 4,
        "xgb_learning_rate": 0.05,
        "use_blend": "false",
        "tabpfn_n_estimators": 8,
        "tabpfn_softmax_temperature": 0.9,
        "chain_variants_config": "config/chain_variants.yaml",
    }
    trial = _make_trial(params)
    args = sweep_runner.build_trial_args(trial, search_space, "^GSPC", [])
    args_str = " ".join(args)
    assert "--use_gkg_change_detector" in args_str
    assert "--gkg_change_model tabpfn" in args_str
    assert "--gkg_tabpfn_n_estimators 12" in args_str


# ---------------------------------------------------------------------------
# read_robust_score_from_mlflow
# ---------------------------------------------------------------------------

def test_read_robust_score_returns_score():
    mock_runs = pd.DataFrame([{"metrics.robust_score": 0.72}])
    with patch("sweep_runner.mlflow") as mock_mlflow:
        mock_mlflow.search_runs.return_value = mock_runs
        score = sweep_runner.read_robust_score_from_mlflow(
            "walk_forward_sweep", "^GSPC", 3, "sweep_GSPC_20260315"
        )
    assert score == pytest.approx(0.72)


def test_read_robust_score_returns_neg_inf_on_empty():
    with patch("sweep_runner.mlflow") as mock_mlflow:
        mock_mlflow.search_runs.return_value = pd.DataFrame()
        score = sweep_runner.read_robust_score_from_mlflow(
            "walk_forward_sweep", "^GSPC", 3, "sweep_GSPC_20260315"
        )
    assert score == float("-inf")


def test_read_robust_score_returns_neg_inf_on_nan():
    mock_runs = pd.DataFrame([{"metrics.robust_score": float("nan")}])
    with patch("sweep_runner.mlflow") as mock_mlflow:
        mock_mlflow.search_runs.return_value = mock_runs
        score = sweep_runner.read_robust_score_from_mlflow(
            "walk_forward_sweep", "^GSPC", 3, "sweep_GSPC_20260315"
        )
    assert score == float("-inf")
