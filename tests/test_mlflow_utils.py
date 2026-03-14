"""Tests for src/quant_risk/tracking/mlflow_utils.py"""
from __future__ import annotations

from quant_risk.tracking.mlflow_utils import MlflowConfig, get_git_sha, is_enabled


def test_mlflow_config_defaults():
    cfg = MlflowConfig()
    assert cfg.enabled is False
    assert cfg.tracking_uri == "file:./mlruns"
    assert cfg.experiment_name == "walk_forward_chain_tab"
    assert cfg.parent_run_name == ""
    assert cfg.strict is False
    assert cfg.extra_tags == {}


def test_is_enabled_false_by_default():
    cfg = MlflowConfig()
    assert is_enabled(cfg) is False


def test_is_enabled_true_when_set():
    from unittest.mock import patch
    cfg = MlflowConfig(enabled=True)
    with patch("quant_risk.tracking.mlflow_utils._MLFLOW_AVAILABLE", True):
        assert is_enabled(cfg) is True


def test_get_git_sha_returns_string():
    sha = get_git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0
