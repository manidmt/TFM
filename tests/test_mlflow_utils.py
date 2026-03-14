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


import pytest
from unittest.mock import MagicMock, patch


def test_log_params_safe_swallows_errors_when_not_strict():
    cfg = MlflowConfig(enabled=True, strict=False)
    with patch("quant_risk.tracking.mlflow_utils._MLFLOW_AVAILABLE", True):
        with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_params.side_effect = RuntimeError("server down")
            # Should not raise
            from quant_risk.tracking.mlflow_utils import log_params_safe
            log_params_safe(cfg, {"horizon": 5})


def test_log_params_safe_raises_in_strict_mode():
    cfg = MlflowConfig(enabled=True, strict=True)
    with patch("quant_risk.tracking.mlflow_utils._MLFLOW_AVAILABLE", True):
        with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_params.side_effect = RuntimeError("server down")
            from quant_risk.tracking.mlflow_utils import log_params_safe
            with pytest.raises(RuntimeError):
                log_params_safe(cfg, {"horizon": 5})


def test_log_params_safe_is_noop_when_disabled():
    cfg = MlflowConfig(enabled=False)
    with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
        from quant_risk.tracking.mlflow_utils import log_params_safe
        log_params_safe(cfg, {"horizon": 5})
        mock_mlflow.log_params.assert_not_called()


def test_log_metrics_safe_swallows_errors_when_not_strict():
    cfg = MlflowConfig(enabled=True, strict=False)
    with patch("quant_risk.tracking.mlflow_utils._MLFLOW_AVAILABLE", True):
        with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.log_metrics.side_effect = RuntimeError("server down")
            from quant_risk.tracking.mlflow_utils import log_metrics_safe
            log_metrics_safe(cfg, {"robust_score": 0.12})


def test_log_metrics_safe_skips_non_finite():
    import math
    cfg = MlflowConfig(enabled=True, strict=False)
    with patch("quant_risk.tracking.mlflow_utils._MLFLOW_AVAILABLE", True):
        with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
            from quant_risk.tracking.mlflow_utils import log_metrics_safe
            log_metrics_safe(cfg, {"a": float("nan"), "b": float("inf"), "c": 0.5})
            called_metrics = mock_mlflow.log_metrics.call_args[0][0]
            assert "a" not in called_metrics
            assert "b" not in called_metrics
            assert called_metrics["c"] == pytest.approx(0.5)


def test_log_artifact_safe_warns_on_missing_file():
    cfg = MlflowConfig(enabled=True, strict=False)
    with patch("quant_risk.tracking.mlflow_utils._MLFLOW_AVAILABLE", True):
        with patch("quant_risk.tracking.mlflow_utils.mlflow"):
            import warnings
            from quant_risk.tracking.mlflow_utils import log_artifact_safe
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                log_artifact_safe(cfg, "/nonexistent/path/file.csv")
            assert any("artifact not found" in str(x.message) for x in w)


import json
import tempfile
from pathlib import Path


def test_log_dict_artifact_writes_correct_filename():
    cfg = MlflowConfig(enabled=True, strict=False)
    # Capture path AND content inside the side_effect while the temp dir still exists.
    captured: dict = {}
    with patch("quant_risk.tracking.mlflow_utils._MLFLOW_AVAILABLE", True):
        with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
            def capture_artifact(path, **kwargs):
                captured["path"] = path
                captured["content"] = json.loads(Path(path).read_text())
            mock_mlflow.log_artifact.side_effect = capture_artifact
            from quant_risk.tracking.mlflow_utils import log_dict_artifact
            log_dict_artifact(cfg, {"key": "value"}, "run_context.json")
    assert captured["path"].endswith("run_context.json")
    assert captured["content"]["key"] == "value"


def test_log_dict_artifact_is_noop_when_disabled():
    cfg = MlflowConfig(enabled=False)
    with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
        from quant_risk.tracking.mlflow_utils import log_dict_artifact
        log_dict_artifact(cfg, {"key": "value"}, "test.json")
        mock_mlflow.log_artifact.assert_not_called()


def test_end_run_safe_is_noop_when_disabled():
    cfg = MlflowConfig(enabled=False)
    with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
        from quant_risk.tracking.mlflow_utils import end_run_safe
        end_run_safe(cfg)
        mock_mlflow.end_run.assert_not_called()


def test_end_run_safe_swallows_errors():
    cfg = MlflowConfig(enabled=True, strict=False)
    with patch("quant_risk.tracking.mlflow_utils._MLFLOW_AVAILABLE", True):
        with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.end_run.side_effect = RuntimeError("already ended")
            from quant_risk.tracking.mlflow_utils import end_run_safe
            end_run_safe(cfg)  # should not raise
