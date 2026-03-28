'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Tests for src/quant_risk/tracking/mlflow_utils.py.
'''
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


def test_start_parent_run_is_noop_when_disabled():
    cfg = MlflowConfig(enabled=False)
    with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
        from quant_risk.tracking.mlflow_utils import start_parent_run
        result = start_parent_run(cfg, run_name="test_run", tags={})
        assert result is None
        mock_mlflow.start_run.assert_not_called()


def test_start_parent_run_sets_experiment_and_uri():
    cfg = MlflowConfig(enabled=True, tracking_uri="file:./test_mlruns", experiment_name="my_exp")
    with patch("quant_risk.tracking.mlflow_utils._MLFLOW_AVAILABLE", True):
        with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value = MagicMock()
            from quant_risk.tracking.mlflow_utils import start_parent_run
            start_parent_run(cfg, run_name="my_run", tags={"k": "v"})
            mock_mlflow.set_tracking_uri.assert_called_once_with("file:./test_mlruns")
            mock_mlflow.set_experiment.assert_called_once_with("my_exp")
            mock_mlflow.start_run.assert_called_once_with(
                run_name="my_run", tags={"k": "v"}
            )


def test_start_child_run_is_noop_when_disabled():
    cfg = MlflowConfig(enabled=False)
    with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
        from quant_risk.tracking.mlflow_utils import start_child_run
        result = start_child_run(cfg, run_name="child", tags={})
        assert result is None
        mock_mlflow.start_run.assert_not_called()


def test_start_child_run_uses_nested_flag():
    cfg = MlflowConfig(enabled=True)
    with patch("quant_risk.tracking.mlflow_utils._MLFLOW_AVAILABLE", True):
        with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value = MagicMock()
            from quant_risk.tracking.mlflow_utils import start_child_run
            start_child_run(cfg, run_name="asset=GSPC", tags={"asset": "GSPC"})
            mock_mlflow.start_run.assert_called_once_with(
                run_name="asset=GSPC", nested=True, tags={"asset": "GSPC"}
            )


import duckdb


def test_read_latest_gkg_manifest_returns_none_when_table_missing():
    # NamedTemporaryFile is deleted after the `with` block; DuckDB creates a new
    # empty database at that path on connect, so gkg_ingest_runs will be absent.
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as f:
        db_path = f.name
    from quant_risk.tracking.mlflow_utils import read_latest_gkg_manifest
    result = read_latest_gkg_manifest(db_path)
    assert result is None


def test_read_latest_gkg_manifest_returns_latest_row():
    import duckdb
    from datetime import datetime
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE gkg_ingest_runs (
                run_ts TIMESTAMP NOT NULL,
                profile_name TEXT,
                profile_version TEXT,
                profile_mode TEXT,
                start_date DATE,
                end_date DATE,
                topic_ids TEXT,
                topics_hash TEXT,
                store_raw BOOLEAN,
                sample_interval_minutes INTEGER,
                max_files_per_day INTEGER,
                max_files_total INTEGER,
                publication_lag_bdays INTEGER,
                request_pause_seconds DOUBLE,
                timeout_seconds INTEGER,
                max_retries INTEGER,
                force_refresh_existing BOOLEAN,
                raw_table TEXT,
                daily_table TEXT,
                inserted_raw_rows BIGINT,
                inserted_daily_rows BIGINT,
                error_count INTEGER
            )
        """)
        con.execute("""
            INSERT INTO gkg_ingest_runs
            (run_ts, profile_name, profile_version, profile_mode, raw_table, daily_table,
             start_date, end_date, topic_ids, topics_hash, store_raw, sample_interval_minutes,
             max_files_per_day, max_files_total, publication_lag_bdays, request_pause_seconds,
             timeout_seconds, max_retries, force_refresh_existing,
             inserted_raw_rows, inserted_daily_rows, error_count)
            VALUES
            ('2026-01-01 10:00:00', 'gkg_v1_light', '2026-03-13', 'light_daily',
             'gdelt_gkg_raw', 'news_features_daily',
             '2020-01-01', '2026-01-01', 'macro_us|fed_inflation', 'abc123', false, 180,
             8, 0, 1, 1.0, 45, 8, false, 100, 50, 0),
            ('2026-02-01 10:00:00', 'gkg_v1_light', '2026-03-13', 'light_daily',
             'gdelt_gkg_raw', 'news_features_daily',
             '2020-01-01', '2026-02-01', 'macro_us|fed_inflation', 'abc123', false, 180,
             8, 0, 1, 1.0, 45, 8, false, 200, 100, 0)
        """)
        con.close()
        from quant_risk.tracking.mlflow_utils import read_latest_gkg_manifest
        result = read_latest_gkg_manifest(str(db_path))
    assert result is not None
    assert result["profile_name"] == "gkg_v1_light"
    # Should return the latest row (2026-02-01)
    assert result["inserted_daily_rows"] == 100


def test_build_run_context_contains_expected_keys():
    import argparse
    from quant_risk.tracking.mlflow_utils import build_run_context
    args = argparse.Namespace(horizon=5, tabular_model="xgb", outdir="runs/test")
    result = build_run_context(
        args=args,
        configs={"features": {"split": {}}, "sources": {}},
        git_sha="abc1234",
    )
    assert "git_sha" in result
    assert "args" in result
    assert "configs" in result
    assert result["git_sha"] == "abc1234"
    assert result["args"]["horizon"] == "5"


def test_start_child_run_propagates_extra_tags():
    cfg = MlflowConfig(enabled=True, extra_tags={"sweep_trial": "7", "optuna_study": "sweep_GSPC_20260315"})
    explicit_tags = {"asset": "^GSPC"}

    with patch("quant_risk.tracking.mlflow_utils._MLFLOW_AVAILABLE", True):
        with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
            mock_mlflow.start_run.return_value = MagicMock()
            from quant_risk.tracking.mlflow_utils import start_child_run
            start_child_run(cfg, run_name="asset=^GSPC", tags=explicit_tags)

    call_kwargs = mock_mlflow.start_run.call_args[1]
    merged = call_kwargs["tags"]
    assert merged["asset"] == "^GSPC"
    assert merged["sweep_trial"] == "7"
    assert merged["optuna_study"] == "sweep_GSPC_20260315"
