'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-27

@description: XGBoost multiclass tabular model.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import set_global_seed, to_numpy_x, to_numpy_y


@dataclass(frozen=True)
class XGBConfig:
    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    n_jobs: int = -1
    random_state: int = 42
    eval_metric: str = "mlogloss"
    early_stopping_rounds: int | None = 30
    seed: int = 42


def make_model(cfg: XGBConfig):
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            "xgboost no está instalado. Instala con: poetry add xgboost"
        ) from e

    set_global_seed(cfg.seed, use_torch=False)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
        eval_metric=cfg.eval_metric,
        early_stopping_rounds=cfg.early_stopping_rounds,
        device="cpu",
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
    x_train = to_numpy_x(X_train, dtype=np.float32)
    y_train_np = to_numpy_y(y_train)
    cfg = getattr(model, "_quant_risk_cfg", XGBConfig())

    if X_valid is not None and y_valid is not None:
        x_valid = to_numpy_x(X_valid, dtype=np.float32)
        y_valid_np = to_numpy_y(y_valid)
        model.fit(x_train, y_train_np, eval_set=[(x_valid, y_valid_np)], verbose=False)
    else:
        model.fit(x_train, y_train_np, verbose=False)

    return model


def predict(model: Any, X) -> np.ndarray:
    preds = model.predict(to_numpy_x(X, dtype=np.float32))
    return np.asarray(preds, dtype=int)


def predict_proba(model: Any, X) -> np.ndarray:
    return np.asarray(model.predict_proba(to_numpy_x(X, dtype=np.float32)), dtype=float)
