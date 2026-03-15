"""MLflow tracking utilities for walk_forward_chain_tab.py.

All public functions are no-ops when MlflowConfig.enabled is False,
so callers need no guard logic of their own.
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import mlflow  # noqa: F401
    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
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
        return mlflow.start_run(run_name=run_name, nested=True, tags=tags or {})
    except Exception as exc:
        _warn_or_raise(cfg, exc)
        return None


def end_run_safe(cfg: MlflowConfig, status: str = "FINISHED") -> None:
    """End the active MLflow run; no-op when tracking is disabled."""
    if not is_enabled(cfg):
        return
    try:
        mlflow.end_run(status=status)
    except Exception as exc:
        _warn_or_raise(cfg, exc)


def log_params_safe(cfg: MlflowConfig, params: dict[str, Any]) -> None:
    """Log a dict of params; values are coerced to str and truncated to 250 chars."""
    if not is_enabled(cfg):
        return
    try:
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
        clean = {
            str(k): float(v)
            for k, v in metrics.items()
            if v is not None and _is_finite_float(v)
        }
        if clean:
            mlflow.log_metrics(clean)
    except Exception as exc:
        _warn_or_raise(cfg, exc)


def log_artifact_safe(cfg: MlflowConfig, path) -> None:
    """Log an existing file as an MLflow artifact; warns if file not found."""
    if not is_enabled(cfg):
        return
    try:
        p = Path(path)
        if p.exists():
            mlflow.log_artifact(str(p))
        else:
            _warn_or_raise(cfg, FileNotFoundError(f"[mlflow] artifact not found: {p}"))
    except Exception as exc:
        _warn_or_raise(cfg, exc)


def log_dict_artifact(cfg: MlflowConfig, data: dict[str, Any], filename: str) -> None:
    """Serialize *data* to a temp file named *filename* and log it as an artifact."""
    if not is_enabled(cfg):
        return
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / filename
            fpath.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(fpath))
    except Exception as exc:
        _warn_or_raise(cfg, exc)


def read_latest_gkg_manifest(db_path) -> dict[str, Any] | None:
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


def _is_finite_float(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False
