'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-04

@description: TabPFN multiclass tabular model.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import set_global_seed, to_numpy_x, to_numpy_y

_TABPFN_MAX_ABS = 1e6


@dataclass(frozen=True)
class TabPFNConfig:
    n_estimators: int = 8
    softmax_temperature: float = 0.9
    balance_probabilities: bool = False
    average_before_softmax: bool = False
    model_path: str = "auto"
    device: str = "auto"
    ignore_pretraining_limits: bool = False
    inference_precision: str = "auto"
    fit_mode: str = "fit_preprocessors"
    memory_saving_mode: str = "auto"
    random_state: int = 0
    n_preprocessing_jobs: int = 1
    seed: int = 42


def _sanitize_tabpfn_array(
    X: Any,
    *,
    stats: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Ensure finite float32 inputs for TabPFN.

    - Replace NaN/Inf using train-derived per-column medians.
    - Clip extreme values per column to robust train-derived bounds.
    """
    arr = to_numpy_x(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    n_features = int(arr.shape[1])

    def _fit_stats(a: np.ndarray) -> dict[str, np.ndarray]:
        medians = np.zeros(n_features, dtype=np.float64)
        clip_abs = np.ones(n_features, dtype=np.float64)
        for j in range(n_features):
            col = a[:, j]
            finite_col = col[np.isfinite(col)]
            if finite_col.size == 0:
                medians[j] = 0.0
                clip_abs[j] = 1.0
                continue
            med = float(np.median(finite_col))
            q01, q99 = np.percentile(finite_col, [1.0, 99.0])
            robust_abs = max(abs(float(q01)), abs(float(q99)), 1.0)
            medians[j] = med
            clip_abs[j] = min(_TABPFN_MAX_ABS, robust_abs * 10.0)
        return {"medians": medians, "clip_abs": clip_abs}

    if stats is None:
        stats = _fit_stats(arr)
    else:
        medians = np.asarray(stats.get("medians"), dtype=np.float64)
        clip_abs = np.asarray(stats.get("clip_abs"), dtype=np.float64)
        if medians.shape != (n_features,) or clip_abs.shape != (n_features,):
            raise ValueError(
                "Shape mismatch between provided sanitize stats and input features: "
                f"expected ({n_features},), got medians.shape={medians.shape}, "
                f"clip_abs.shape={clip_abs.shape}"
            )

    medians = np.asarray(stats["medians"], dtype=np.float64)
    clip_abs = np.asarray(stats["clip_abs"], dtype=np.float64)

    bad = ~np.isfinite(arr)
    if bad.any():
        arr = arr.copy()
        _, cols = np.where(bad)
        arr[bad] = medians[cols]

    arr = np.clip(arr, -clip_abs, clip_abs)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr, {"medians": medians, "clip_abs": clip_abs}


def make_model(cfg: TabPFNConfig):
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as e:
        raise ImportError(
            "tabpfn no está instalado. Instala con: poetry add tabpfn"
        ) from e

    set_global_seed(cfg.seed, use_torch=True)

    model = TabPFNClassifier(
        n_estimators=cfg.n_estimators,
        softmax_temperature=cfg.softmax_temperature,
        balance_probabilities=cfg.balance_probabilities,
        average_before_softmax=cfg.average_before_softmax,
        model_path=cfg.model_path,
        device=cfg.device,
        ignore_pretraining_limits=cfg.ignore_pretraining_limits,
        inference_precision=cfg.inference_precision,
        fit_mode=cfg.fit_mode,
        memory_saving_mode=cfg.memory_saving_mode,
        random_state=cfg.random_state,
        n_preprocessing_jobs=cfg.n_preprocessing_jobs,
    )
    model._quant_risk_cfg = cfg
    return model


def fit(
    model: Any,
    X_train,
    y_train,
    X_valid=None,
    y_valid=None,
):
    _ = (X_valid, y_valid)
    x_train, stats = _sanitize_tabpfn_array(X_train)
    y_train_np = to_numpy_y(y_train)
    model.fit(x_train, y_train_np)
    model._quant_risk_sanitize_stats = stats
    return model


def predict(model: Any, X) -> np.ndarray:
    stats = getattr(model, "_quant_risk_sanitize_stats", None)
    x, _ = _sanitize_tabpfn_array(X, stats=stats)
    preds = model.predict(x)
    return np.asarray(preds, dtype=int)


def predict_proba(model: Any, X) -> np.ndarray:
    stats = getattr(model, "_quant_risk_sanitize_stats", None)
    x, _ = _sanitize_tabpfn_array(X, stats=stats)
    return np.asarray(model.predict_proba(x), dtype=float)
