# MLflow v1 Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in MLflow experiment tracking to `walk_forward_chain_tab.py` via a clean utility module, without touching any existing output logic.

**Architecture:** A new `src/quant_risk/tracking/mlflow_utils.py` module provides all MLflow primitives (lazy import, safe wrappers, manifest reader). `walk_forward_chain_tab.py` calls these helpers at six well-defined integration points. All existing CSV/JSON filesystem outputs remain unchanged; MLflow logs them *additionally* as artifacts.

**Tech Stack:** Python 3.10+, `mlflow ^2.13`, `duckdb` (already a dependency), `unittest.mock` for tests.

---

## Chunk 1: Tracking module (TDD)

### Task 1: Add mlflow dependency and create module skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/quant_risk/tracking/__init__.py`
- Create: `src/quant_risk/tracking/mlflow_utils.py` (skeleton only)

- [ ] **Step 1: Add mlflow to pyproject.toml**

In `pyproject.toml`, add under `[tool.poetry.dependencies]`:
```toml
mlflow = "^2.13"
```

- [ ] **Step 2: Install the new dependency**

```bash
poetry add mlflow@^2.13
```
Expected: resolves and updates `poetry.lock`.

- [ ] **Step 3: Create `src/quant_risk/tracking/__init__.py`**

```python
from quant_risk.tracking.mlflow_utils import (
    MlflowConfig,
    is_enabled,
    get_git_sha,
    start_parent_run,
    start_child_run,
    end_run_safe,
    log_params_safe,
    log_metrics_safe,
    log_artifact_safe,
    log_dict_artifact,
    read_latest_gkg_manifest,
    build_run_context,
)

__all__ = [
    "MlflowConfig",
    "is_enabled",
    "get_git_sha",
    "start_parent_run",
    "start_child_run",
    "end_run_safe",
    "log_params_safe",
    "log_metrics_safe",
    "log_artifact_safe",
    "log_dict_artifact",
    "read_latest_gkg_manifest",
    "build_run_context",
]
```

- [ ] **Step 4: Create `src/quant_risk/tracking/mlflow_utils.py` as an empty skeleton**

```python
"""MLflow tracking utilities for walk_forward_chain_tab.py.

All public functions are no-ops when MlflowConfig.enabled is False,
so callers need no guard logic of their own.
"""
from __future__ import annotations
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml poetry.lock src/quant_risk/tracking/
git commit -m "feat: add mlflow dependency and tracking module skeleton"
```

---

### Task 2: MlflowConfig dataclass + is_enabled + get_git_sha (TDD)

**Files:**
- Modify: `src/quant_risk/tracking/mlflow_utils.py`
- Create: `tests/test_mlflow_utils.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_mlflow_utils.py`:

```python
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
    cfg = MlflowConfig(enabled=True)
    # mlflow is installed in this env, so this should return True
    assert is_enabled(cfg) is True


def test_get_git_sha_returns_string():
    sha = get_git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_mlflow_utils.py -v
```
Expected: `ImportError` — `MlflowConfig` not yet defined.

- [ ] **Step 3: Implement MlflowConfig, is_enabled, get_git_sha in mlflow_utils.py**

```python
from __future__ import annotations

import subprocess
import warnings
from dataclasses import dataclass, field
from typing import Any

try:
    import mlflow as _mlflow  # noqa: F401
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


@dataclass
class MlflowConfig:
    """Configuration for MLflow tracking in walk_forward_chain_tab.py."""

    enabled: bool = False
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "walk_forward_chain_tab"
    parent_run_name: str = ""
    strict: bool = False
    extra_tags: dict[str, str] = field(default_factory=dict)


def is_enabled(cfg: MlflowConfig) -> bool:
    """Return True only when tracking is requested AND mlflow is importable."""
    return cfg.enabled and _MLFLOW_AVAILABLE


def get_git_sha() -> str:
    """Return the short HEAD git SHA, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _warn_or_raise(cfg: MlflowConfig, exc: Exception) -> None:
    if cfg.strict:
        raise exc
    warnings.warn(f"[mlflow] {exc}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_mlflow_utils.py::test_mlflow_config_defaults tests/test_mlflow_utils.py::test_is_enabled_false_by_default tests/test_mlflow_utils.py::test_is_enabled_true_when_set tests/test_mlflow_utils.py::test_get_git_sha_returns_string -v
```
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_risk/tracking/mlflow_utils.py tests/test_mlflow_utils.py
git commit -m "feat: MlflowConfig dataclass, is_enabled, get_git_sha"
```

---

### Task 3: Safe logging helpers (TDD)

**Files:**
- Modify: `src/quant_risk/tracking/mlflow_utils.py`
- Modify: `tests/test_mlflow_utils.py`

- [ ] **Step 1: Add failing tests for safe logging helpers**

Append to `tests/test_mlflow_utils.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_mlflow_utils.py -k "log_params or log_metrics or log_artifact" -v
```
Expected: `ImportError` — functions not yet defined.

- [ ] **Step 3: Implement log_params_safe, log_metrics_safe, log_artifact_safe**

Append to `mlflow_utils.py` after `_warn_or_raise`:

```python
import math
from pathlib import Path


def log_params_safe(cfg: MlflowConfig, params: dict[str, Any]) -> None:
    """Log a dict of params; values are coerced to str and truncated to 250 chars."""
    if not is_enabled(cfg):
        return
    try:
        import mlflow
        str_params = {
            str(k): str(v)[:250]
            for k, v in params.items()
            if v is not None
        }
        if str_params:
            mlflow.log_params(str_params)
    except Exception as exc:
        _warn_or_raise(cfg, exc)


def log_metrics_safe(cfg: MlflowConfig, metrics: dict[str, Any]) -> None:
    """Log a dict of float metrics; NaN/Inf values are silently skipped."""
    if not is_enabled(cfg):
        return
    try:
        import mlflow
        clean = {
            str(k): float(v)
            for k, v in metrics.items()
            if v is not None and _is_finite_float(v)
        }
        if clean:
            mlflow.log_metrics(clean)
    except Exception as exc:
        _warn_or_raise(cfg, exc)


def log_artifact_safe(cfg: MlflowConfig, path: str | Path) -> None:
    """Log an existing file as an MLflow artifact; warns if file not found."""
    if not is_enabled(cfg):
        return
    try:
        import mlflow
        p = Path(path)
        if p.exists():
            mlflow.log_artifact(str(p))
        else:
            warnings.warn(f"[mlflow] artifact not found: {p}")
    except Exception as exc:
        _warn_or_raise(cfg, exc)


def _is_finite_float(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_mlflow_utils.py -k "log_params or log_metrics or log_artifact" -v
```
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_risk/tracking/mlflow_utils.py tests/test_mlflow_utils.py
git commit -m "feat: log_params_safe, log_metrics_safe, log_artifact_safe"
```

---

### Task 4: log_dict_artifact + end_run_safe (TDD)

**Files:**
- Modify: `src/quant_risk/tracking/mlflow_utils.py`
- Modify: `tests/test_mlflow_utils.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_mlflow_utils.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_mlflow_utils.py -k "dict_artifact or end_run" -v
```
Expected: `ImportError` — functions not yet defined.

- [ ] **Step 3: Implement log_dict_artifact and end_run_safe**

Append to `mlflow_utils.py`:

```python
import json
import tempfile


def log_dict_artifact(cfg: MlflowConfig, data: dict[str, Any], filename: str) -> None:
    """Serialize *data* to a temp file named *filename* and log it as an artifact."""
    if not is_enabled(cfg):
        return
    try:
        import mlflow
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / filename
            fpath.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(fpath))
    except Exception as exc:
        _warn_or_raise(cfg, exc)


def end_run_safe(cfg: MlflowConfig) -> None:
    """End the active MLflow run; no-op when tracking is disabled."""
    if not is_enabled(cfg):
        return
    try:
        import mlflow
        mlflow.end_run()
    except Exception as exc:
        _warn_or_raise(cfg, exc)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_mlflow_utils.py -k "dict_artifact or end_run" -v
```
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_risk/tracking/mlflow_utils.py tests/test_mlflow_utils.py
git commit -m "feat: log_dict_artifact and end_run_safe"
```

---

### Task 5: start_parent_run + start_child_run (TDD)

**Files:**
- Modify: `src/quant_risk/tracking/mlflow_utils.py`
- Modify: `tests/test_mlflow_utils.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_mlflow_utils.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_mlflow_utils.py -k "start_parent or start_child" -v
```
Expected: `ImportError` — functions not yet defined.

- [ ] **Step 3: Implement start_parent_run and start_child_run**

Append to `mlflow_utils.py`:

```python
def start_parent_run(
    cfg: MlflowConfig,
    run_name: str = "",
    tags: dict[str, str] | None = None,
) -> Any:
    """Configure the MLflow experiment and start the parent run.

    Returns the active MLflow run object, or None when tracking is disabled.
    """
    if not is_enabled(cfg):
        return None
    try:
        import mlflow
        mlflow.set_tracking_uri(cfg.tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)
        merged_tags = {**(tags or {}), **cfg.extra_tags}
        return mlflow.start_run(run_name=run_name, tags=merged_tags)
    except Exception as exc:
        _warn_or_raise(cfg, exc)
        return None


def start_child_run(
    cfg: MlflowConfig,
    run_name: str,
    tags: dict[str, str] | None = None,
) -> Any:
    """Start a nested child run inside the active parent run.

    Returns the active MLflow run object, or None when tracking is disabled.
    """
    if not is_enabled(cfg):
        return None
    try:
        import mlflow
        return mlflow.start_run(run_name=run_name, nested=True, tags=tags or {})
    except Exception as exc:
        _warn_or_raise(cfg, exc)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_mlflow_utils.py -k "start_parent or start_child" -v
```
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_risk/tracking/mlflow_utils.py tests/test_mlflow_utils.py
git commit -m "feat: start_parent_run and start_child_run"
```

---

### Task 6: read_latest_gkg_manifest + build_run_context (TDD)

**Files:**
- Modify: `src/quant_risk/tracking/mlflow_utils.py`
- Modify: `tests/test_mlflow_utils.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_mlflow_utils.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_mlflow_utils.py -k "gkg_manifest or run_context" -v
```
Expected: `ImportError` — functions not yet defined.

- [ ] **Step 3: Implement read_latest_gkg_manifest and build_run_context**

Append to `mlflow_utils.py`:

```python
def read_latest_gkg_manifest(db_path: str | Path) -> dict[str, Any] | None:
    """Query the latest row from gkg_ingest_runs in DuckDB.

    Returns a dict of manifest fields, or None if the table does not exist
    or the database is empty.
    """
    try:
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            df = con.execute(
                "SELECT * FROM gkg_ingest_runs ORDER BY run_ts DESC LIMIT 1"
            ).fetchdf()
        finally:
            con.close()
        if df.empty:
            return None
        return {k: (None if _is_nat(v) else v) for k, v in df.iloc[0].to_dict().items()}
    except Exception as exc:
        warnings.warn(f"[mlflow] Could not read gkg_ingest_runs: {exc}")
        return None


def _is_nat(v: Any) -> bool:
    """Return True for pandas NaT values (not JSON serializable)."""
    try:
        import pandas as pd
        return pd.isna(v)
    except Exception:
        return False


def build_run_context(
    args: Any,
    configs: dict[str, Any],
    git_sha: str,
) -> dict[str, Any]:
    """Build a JSON-serialisable run-context snapshot from CLI args and loaded configs."""
    return {
        "git_sha": git_sha,
        "args": {k: str(v) for k, v in vars(args).items()},
        "configs": configs,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_mlflow_utils.py -k "gkg_manifest or run_context" -v
```
Expected: 3 PASS.

- [ ] **Step 5: Run the full test_mlflow_utils.py suite**

```bash
poetry run pytest tests/test_mlflow_utils.py -v
```
Expected: all tests PASS.

- [ ] **Step 6: Run the full existing test suite to confirm no regressions**

```bash
poetry run pytest -q
```
Expected: all existing tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/quant_risk/tracking/mlflow_utils.py tests/test_mlflow_utils.py
git commit -m "feat: read_latest_gkg_manifest and build_run_context"
```

---

## Chunk 2: CLI flags + walk_forward integration

### Task 7: Add CLI flags + build MlflowConfig in main()

**Files:**
- Modify: `scripts/walk_forward_chain_tab.py`

The 6 new flags slot in after the existing `--grid_profile` flag (line ~2781). The `MlflowConfig` is built immediately after `args = parser.parse_args()`.

- [ ] **Step 1: Add import at the top of walk_forward_chain_tab.py**

Find the existing imports block (around line 30). After the last `from quant_risk...` import, add:

```python
from quant_risk.tracking import mlflow_utils as tracking
from quant_risk.tracking.mlflow_utils import MlflowConfig
```

- [ ] **Step 2: Add CLI flags**

Find the line `args = parser.parse_args()` (around line 2782). **Before** that line, add these 6 arguments after `--grid_profile`:

```python
# --- MLflow tracking ---
parser.add_argument(
    "--use_mlflow",
    action="store_true",
    help="Enable MLflow experiment tracking.",
)
parser.add_argument(
    "--mlflow_tracking_uri",
    default="file:./mlruns",
    help="MLflow tracking store URI.",
)
parser.add_argument(
    "--mlflow_experiment",
    default="walk_forward_chain_tab",
    help="MLflow experiment name.",
)
parser.add_argument(
    "--mlflow_parent_run_name",
    default="",
    help="Override parent run name (auto-generated timestamp if empty).",
)
parser.add_argument(
    "--mlflow_strict",
    action="store_true",
    help="Abort the run if any MLflow tracking call fails.",
)
parser.add_argument(
    "--mlflow_tags_json",
    default="",
    help="Extra MLflow tags as a JSON string, e.g. '{\"env\": \"dev\"}'.",
)
```

- [ ] **Step 3: Build MlflowConfig after args = parser.parse_args()**

Add this block immediately after `args = parser.parse_args()`:

```python
# --- Build MLflow config ---
_mlflow_extra_tags: dict[str, str] = {}
if args.mlflow_tags_json:
    try:
        _mlflow_extra_tags = json.loads(args.mlflow_tags_json)
    except json.JSONDecodeError as _e:
        raise SystemExit(f"Invalid --mlflow_tags_json: {_e}") from _e
mlflow_cfg = MlflowConfig(
    enabled=bool(args.use_mlflow),
    tracking_uri=str(args.mlflow_tracking_uri),
    experiment_name=str(args.mlflow_experiment),
    parent_run_name=str(args.mlflow_parent_run_name),
    strict=bool(args.mlflow_strict),
    extra_tags=_mlflow_extra_tags,
)
```

- [ ] **Step 4: Verify the script still parses without errors**

```bash
poetry run python scripts/walk_forward_chain_tab.py --help
```
Expected: help text shows new `--use_mlflow`, `--mlflow_tracking_uri`, `--mlflow_experiment`, `--mlflow_parent_run_name`, `--mlflow_strict`, `--mlflow_tags_json` flags. No errors.

- [ ] **Step 5: Run existing tests to confirm no regressions**

```bash
poetry run pytest -q
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/walk_forward_chain_tab.py
git commit -m "feat: add MLflow CLI flags and MlflowConfig construction to walk_forward"
```

---

### Task 8: Helper param/metric builders + parent run lifecycle

**Files:**
- Modify: `scripts/walk_forward_chain_tab.py`

All helpers are private functions (`_build_mlflow_parent_tags`, `_build_mlflow_child_params`, `_build_mlflow_child_metrics`) defined just above `main()`. They are standalone and easy to unit-test via `importlib.util`.

- [ ] **Step 1: Add _build_mlflow_parent_tags helper above main()**

Find the `def main()` line (~2608). Insert these helpers just before it:

```python
def _build_mlflow_parent_tags(args: Any, git_sha: str, sources_cfg: dict) -> dict[str, str]:
    """Build tags for the parent MLflow run from CLI args and loaded configs."""
    gkg_cfg = sources_cfg.get("gkg", {})
    return {
        "script": "walk_forward_chain_tab.py",
        "component": "walk_forward",
        "git_sha": git_sha,
        "horizon": str(args.horizon),
        "tabular_model": str(args.tabular_model),
        "news_profile": str(gkg_cfg.get("profile_name", "")),
        "news_profile_version": str(gkg_cfg.get("profile_version", "")),
    }


def _build_mlflow_child_tags(
    asset_name: str,
    group_tickers: tuple[str, ...],
    args: Any,
) -> dict[str, str]:
    """Build tags for one child (asset) MLflow run."""
    return {
        "asset": str(asset_name),
        "tickers": "|".join(group_tickers),
        "horizon": str(args.horizon),
        "tabular_model": str(args.tabular_model),
        "use_gkg_change_detector": str(bool(args.use_gkg_change_detector)),
        "use_gkg_change_gate": str(bool(args.use_gkg_change_gate)),
        "use_gkg_change_alpha": str(bool(args.use_gkg_change_alpha)),
    }


def _build_mlflow_child_params(args: Any, sources_cfg: dict, features_cfg: dict) -> dict[str, Any]:
    """Build the full params dict logged to a child MLflow run.

    Covers runner/model params, GKG sidecar params, and data/feature profile params.
    """
    gkg_cfg = sources_cfg.get("gkg", {})
    news_cfg = features_cfg.get("news_features", {})
    return {
        # Runner / model params
        "horizon": args.horizon,
        "tabular_model": args.tabular_model,
        "per_ticker": args.per_ticker,
        "grid_profile": args.grid_profile,
        "min_train_end": args.min_train_end,
        "max_valid_end": args.max_valid_end,
        "valid_months": args.valid_months,
        "step_months": args.step_months,
        "max_struct_configs": args.max_struct_configs,
        "max_model_configs": args.max_model_configs,
        "min_folds_ok": args.min_folds_ok,
        "min_positive_rate": args.min_positive_rate,
        "min_non_transition_delta": args.min_non_transition_delta,
        "min_high_vol_recall_delta": args.min_high_vol_recall_delta,
        "stability_lambda": args.stability_lambda,
        "positive_rate_lambda": args.positive_rate_lambda,
        "non_transition_penalty_lambda": args.non_transition_penalty_lambda,
        "transition_bonus_lambda": args.transition_bonus_lambda,
        "prune_after_folds": args.prune_after_folds,
        "prune_delta_threshold": args.prune_delta_threshold,
        "use_blend": args.use_blend,
        "blend_alphas": args.blend_alphas,
        "blend_conf_betas": args.blend_conf_betas,
        "class_threshold_grid": args.class_threshold_grid,
        "gate_thresholds": args.gate_thresholds,
        "calibration_method": args.calibration_method,
        # GKG sidecar params
        "use_gkg_change_detector": args.use_gkg_change_detector,
        "use_gkg_change_gate": args.use_gkg_change_gate,
        "use_gkg_change_alpha": args.use_gkg_change_alpha,
        "gkg_change_model": args.gkg_change_model,
        "gkg_change_context_cols": args.gkg_change_context_cols,
        "gkg_change_calibration_method": args.gkg_change_calibration_method,
        "gkg_change_gate_thresholds": args.gkg_change_gate_thresholds,
        "gkg_change_alpha_weights": args.gkg_change_alpha_weights,
        "gkg_change_alpha_weights_by_asset": args.gkg_change_alpha_weights_by_asset,
        "gkg_change_blend_hook_weight": args.gkg_change_blend_hook_weight,
        # Data / feature profile params
        "gkg_profile_name": gkg_cfg.get("profile_name", ""),
        "gkg_profile_version": gkg_cfg.get("profile_version", ""),
        "gkg_profile_mode": gkg_cfg.get("profile_mode", ""),
        "gkg_start": gkg_cfg.get("start", ""),
        "gkg_store_raw": gkg_cfg.get("store_raw", False),
        "gkg_sample_interval_minutes": gkg_cfg.get("sample_interval_minutes", ""),
        "gkg_max_files_per_day": gkg_cfg.get("max_files_per_day", ""),
        "gkg_publication_lag_bdays": gkg_cfg.get("publication_lag_bdays", ""),
        "news_feature_profile_name": news_cfg.get("profile_name", ""),
        "news_feature_profile_version": news_cfg.get("profile_version", ""),
        "news_source_table": news_cfg.get("source_table", ""),
        "news_windows": str(news_cfg.get("windows", [])),
        "news_compact_mode": news_cfg.get("compact_mode", False),
        "news_include_interactions": news_cfg.get("include_interactions", False),
    }


def _build_mlflow_child_metrics(best: dict[str, Any], final_cmp: dict[str, Any]) -> dict[str, Any]:
    """Extract the child-run metrics from the selected best row and test comparison."""
    metrics: dict[str, Any] = {
        # Walk-forward selection metrics
        "robust_score": best.get("robust_score"),
        "n_folds_ok": best.get("n_folds_ok"),
        "mean_delta_macro_vs_persistence": best.get("mean_delta_macro_vs_persistence"),
        "mean_delta_transition_macro_f1_vs_persistence": best.get(
            "mean_delta_transition_macro_f1_vs_persistence"
        ),
        "mean_delta_non_transition_macro_f1_vs_persistence": best.get(
            "mean_delta_non_transition_macro_f1_vs_persistence"
        ),
        "mean_delta_high_vol_recall_vs_persistence": best.get(
            "mean_delta_high_vol_recall_vs_persistence"
        ),
        "oof_delta_macro_vs_persistence": best.get("oof_delta_macro_vs_persistence"),
        "selected_struct_id": best.get("struct_id"),
        "selected_model_id": best.get("xgb_id"),
        "selected_blend_alpha": best.get("oof_selected_alpha"),
        "selected_blend_beta": best.get("oof_selected_beta"),
        "selected_gate_threshold": best.get("oof_selected_gate_threshold"),
        "selected_change_gate_threshold": best.get("oof_selected_change_gate_threshold"),
        "selected_change_alpha_weight": best.get("oof_selected_change_alpha_weight"),
    }
    # Test-set metrics from final comparison
    for key in (
        "delta_test_macro_f1_vs_persistence",
        "delta_transition_macro_f1_vs_persistence_test",
        "delta_non_transition_macro_f1_vs_persistence_test",
        "delta_test_high_vol_recall_vs_persistence",
        "gkg_change_macro_f1_test",
        "gkg_change_auc_test",
        "gkg_change_brier_test",
        "change_gating_rate_test",
        "mean_p_change_test",
        "mean_effective_alpha_test",
    ):
        if key in final_cmp:
            metrics[key] = final_cmp[key]
    return metrics
```

- [ ] **Step 2: Add parent run lifecycle to main()**

Find the lines (around line 2867–2870):
```python
    global_best_rows: list[dict[str, Any]] = []
    final_compare_rows: list[dict[str, Any]] = []

    for asset_name, group_tickers in groups:
```

Replace with:
```python
    global_best_rows: list[dict[str, Any]] = []
    final_compare_rows: list[dict[str, Any]] = []

    # --- MLflow: detect git SHA, build parent tags, open parent run ---
    _git_sha = tracking.get_git_sha()
    _parent_tags = _build_mlflow_parent_tags(args, _git_sha, sources_cfg)
    _auto_run_name = (
        args.mlflow_parent_run_name
        or f"wf_chain_tab__h{args.horizon}__{args.tabular_model}__{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )
    _mlflow_parent = tracking.start_parent_run(mlflow_cfg, run_name=_auto_run_name, tags=_parent_tags)
    # Log parent-level params (shared across all assets)
    _child_params = _build_mlflow_child_params(args, sources_cfg, features_cfg)
    tracking.log_params_safe(mlflow_cfg, _child_params)
    # Log config file snapshots as artifacts
    tracking.log_artifact_safe(mlflow_cfg, args.config_features)
    tracking.log_artifact_safe(mlflow_cfg, args.config_sources)
    # Log CLI args snapshot and run context JSON
    tracking.log_dict_artifact(mlflow_cfg, {k: str(v) for k, v in vars(args).items()}, "cli_args.json")
    _run_ctx = tracking.build_run_context(args, {"features": features_cfg, "sources": sources_cfg}, _git_sha)
    tracking.log_dict_artifact(mlflow_cfg, _run_ctx, "run_context.json")

    try:
        for asset_name, group_tickers in groups:
```

- [ ] **Step 3: Close the try block and end parent run after the loop**

Find the lines after the `for asset_name` loop ends (around line 3542):
```python
    root_out = Path(args.outdir) / f"h{args.horizon}"
    if global_best_rows:
        pd.DataFrame(global_best_rows).to_csv(root_out / "best_by_asset.csv", index=False)
        print("\nBest-by-asset:", root_out / "best_by_asset.csv")
    if final_compare_rows and len(groups) > 1:
        pd.DataFrame(final_compare_rows).to_csv(root_out / "final_vs_persistence.csv", index=False)
        print("Final-vs-persistence:", root_out / "final_vs_persistence.csv")
    return 0
```

Replace with:
```python
        root_out = Path(args.outdir) / f"h{args.horizon}"
        if global_best_rows:
            pd.DataFrame(global_best_rows).to_csv(root_out / "best_by_asset.csv", index=False)
            print("\nBest-by-asset:", root_out / "best_by_asset.csv")
        if final_compare_rows and len(groups) > 1:
            pd.DataFrame(final_compare_rows).to_csv(root_out / "final_vs_persistence.csv", index=False)
            print("Final-vs-persistence:", root_out / "final_vs_persistence.csv")
        # --- MLflow: log parent-level aggregates ---
        tracking.log_metrics_safe(mlflow_cfg, {
            "n_asset_groups": float(len(groups)),
            "n_successful_children": float(len(final_compare_rows)),
            "mean_delta_test_macro_f1": float(
                sum(r.get("delta_test_macro_f1_vs_persistence", 0.0) for r in final_compare_rows)
                / len(final_compare_rows)
            ) if final_compare_rows else 0.0,
        })
        if global_best_rows:
            tracking.log_artifact_safe(mlflow_cfg, root_out / "best_by_asset.csv")
        if final_compare_rows and len(groups) > 1:
            tracking.log_artifact_safe(mlflow_cfg, root_out / "final_vs_persistence.csv")
        return 0
    finally:
        tracking.end_run_safe(mlflow_cfg)  # ends parent run
```

> **Important indentation note:** The `root_out = ...` block and everything below it (including `return 0`) must be indented one level inside the `try:` block you opened in Step 2. The `finally:` is at the same indentation as `try:`.

- [ ] **Step 4: Verify the script still runs --help cleanly**

```bash
poetry run python scripts/walk_forward_chain_tab.py --help
```
Expected: exits 0, help text visible.

- [ ] **Step 5: Run existing tests to confirm no regressions**

```bash
poetry run pytest -q
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/walk_forward_chain_tab.py
git commit -m "feat: add parent run lifecycle and helper builders to walk_forward"
```

---

### Task 9: Child run lifecycle per asset

**Files:**
- Modify: `scripts/walk_forward_chain_tab.py`

- [ ] **Step 1: Open child run at the top of each asset iteration**

Find the top of the `for asset_name, group_tickers in groups:` body. The first few lines are (around line 2871):
```python
        asset_gkg_change_alpha_weights = _resolve_asset_specific_grid(
```

Just before that line (right after the `for asset_name` line), add:

```python
        # --- MLflow: open child run for this asset ---
        _child_tags = _build_mlflow_child_tags(asset_name, group_tickers, args)
        _mlflow_child = tracking.start_child_run(
            mlflow_cfg, run_name=f"asset={asset_name}", tags=_child_tags
        )
        try:
```

And indent all the existing asset body one level further inside a `try:` block.

- [ ] **Step 2: Log child params + artifacts + close child run at the end of each asset**

Find the end of the asset loop body — the final lines before the next iteration are (around line 3535–3540):
```python
        print(
            f"Asset={asset_name} | Folds={len(folds)} | Struct={len(structural_cfgs)} | "
            f"{str(args.tabular_model).upper()}={len(xgb_cfgs)}"
        )
        print(summary_df.head(5).to_string(index=False))
        print("\nSaved:", outdir)
```

After these print statements, add (still inside the `try:` for the child run):

```python
            # --- MLflow: log child metrics, artifacts, manifest, close run ---
            _child_metrics = _build_mlflow_child_metrics(best, final_cmp)
            tracking.log_metrics_safe(mlflow_cfg, _child_metrics)
            tracking.log_artifact_safe(mlflow_cfg, outdir / "fold_metrics.csv")
            tracking.log_artifact_safe(mlflow_cfg, outdir / "summary.csv")
            tracking.log_artifact_safe(mlflow_cfg, outdir / "best.json")
            tracking.log_artifact_safe(mlflow_cfg, outdir / "final_vs_persistence.csv")
            # GKG dataset provenance
            if bool(args.use_gkg_change_detector) or sources_cfg.get("gkg", {}).get("enabled", False):
                _manifest = tracking.read_latest_gkg_manifest(db_path)
                if _manifest is not None:
                    _manifest_params = {
                        f"gkg_manifest_{k}": str(v)
                        for k, v in _manifest.items()
                        if k in {
                            "profile_name", "profile_version", "profile_mode",
                            "start_date", "end_date", "topics_hash",
                            "store_raw", "sample_interval_minutes",
                            "max_files_per_day", "publication_lag_bdays",
                            "raw_table", "daily_table",
                        }
                    }
                    tracking.log_params_safe(mlflow_cfg, _manifest_params)
                    tracking.log_dict_artifact(mlflow_cfg, _manifest, "dataset_manifest_gkg.json")
```

Then, close the `try:` for the child run with a `finally:` block at the same indentation:

```python
        finally:
            tracking.end_run_safe(mlflow_cfg)  # ends child run
```

- [ ] **Step 3: Verify the script still runs --help cleanly**

```bash
poetry run python scripts/walk_forward_chain_tab.py --help
```
Expected: exits 0.

- [ ] **Step 4: Run existing tests to confirm no regressions**

```bash
poetry run pytest -q
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/walk_forward_chain_tab.py
git commit -m "feat: add child run lifecycle with metrics, artifacts, and GKG provenance"
```

---

### Task 10: Integration smoke test

**Files:**
- Create: `tests/test_walk_forward_mlflow_smoke.py`

This test uses `importlib.util` (same pattern as `test_chain_gating_calibration_smoke.py`) to load the script as a module and verify: (a) the new helper functions produce correct output without any live MLflow server, and (b) `MlflowConfig` construction from a mock `args` namespace works correctly.

- [ ] **Step 1: Write the tests**

Create `tests/test_walk_forward_mlflow_smoke.py`:

```python
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
```

- [ ] **Step 2: Run the new smoke tests**

```bash
poetry run pytest tests/test_walk_forward_mlflow_smoke.py -v
```
Expected: 5 PASS.

- [ ] **Step 3: Run the full test suite**

```bash
poetry run pytest -q
```
Expected: all tests pass (no regressions anywhere).

- [ ] **Step 4: Commit**

```bash
git add tests/test_walk_forward_mlflow_smoke.py
git commit -m "test: add MLflow smoke tests for walk_forward helper functions"
```

---

### Task 11: Final acceptance check

- [ ] **Step 1: Verify --help still works**

```bash
poetry run python scripts/walk_forward_chain_tab.py --help
```
Expected: exits 0, MLflow flags visible.

- [ ] **Step 2: Verify the script works without --use_mlflow (backward compatibility)**

```bash
poetry run python scripts/walk_forward_chain_tab.py \
  --tickers "^GSPC" \
  --horizon 5 \
  --tabular_model xgb \
  --max_struct_configs 1 \
  --max_model_configs 1 \
  --min_folds_ok 1 \
  --outdir runs/mlflow_v1_smoke_test
```
Expected: runs to completion (or fails only on missing data), no MLflow-related errors.

- [ ] **Step 3: Run the full test suite one final time**

```bash
poetry run pytest -q
```
Expected: all tests pass.

- [ ] **Step 4: Final commit**

```bash
git add -p  # review any unstaged changes
git commit -m "feat: MLflow v1 Phase 1 complete — opt-in tracking for walk_forward_chain_tab"
```
