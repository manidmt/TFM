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
    x_train = to_numpy_x(X_train, dtype=np.float32)
    y_train_np = to_numpy_y(y_train)
    model.fit(x_train, y_train_np)
    return model


def predict(model: Any, X) -> np.ndarray:
    preds = model.predict(to_numpy_x(X, dtype=np.float32))
    return np.asarray(preds, dtype=int)


def predict_proba(model: Any, X) -> np.ndarray:
    return np.asarray(model.predict_proba(to_numpy_x(X, dtype=np.float32)), dtype=float)
