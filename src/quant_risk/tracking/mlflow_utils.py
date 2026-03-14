"""MLflow tracking utilities for walk_forward_chain_tab.py.

All public functions are no-ops when MlflowConfig.enabled is False,
so callers need no guard logic of their own.
"""
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


# ---------------------------------------------------------------------------
# Stubs – to be implemented in subsequent tasks
# ---------------------------------------------------------------------------

def start_parent_run(cfg: MlflowConfig, tags: dict[str, Any] | None = None) -> Any:
    """Start a parent MLflow run. No-op when tracking is disabled."""
    return None


def start_child_run(cfg: MlflowConfig, run_name: str, tags: dict[str, Any] | None = None) -> Any:
    """Start a child MLflow run. No-op when tracking is disabled."""
    return None


def end_run_safe(cfg: MlflowConfig, status: str = "FINISHED") -> None:
    """End the active MLflow run. No-op when tracking is disabled."""
    return None


def log_params_safe(cfg: MlflowConfig, params: dict[str, Any]) -> None:
    """Log parameters to the active MLflow run. No-op when tracking is disabled."""
    return None


def log_metrics_safe(cfg: MlflowConfig, metrics) -> None:
    """Log metrics to the active MLflow run. No-op when tracking is disabled."""
    return None


def log_artifact_safe(cfg: MlflowConfig, path) -> None:
    """Log an artifact to the active MLflow run. No-op when tracking is disabled."""
    return None


def log_dict_artifact(cfg: MlflowConfig, data: dict[str, Any], artifact_name: str) -> None:
    """Log a dict as a JSON artifact. No-op when tracking is disabled."""
    return None


def read_latest_gkg_manifest(manifest_path: str) -> dict[str, Any]:
    """Read the latest GKG manifest file and return its contents."""
    return {}


def build_run_context(cfg: MlflowConfig, **kwargs: Any) -> dict[str, Any]:
    """Build a run context dict for tagging MLflow runs."""
    return {}
