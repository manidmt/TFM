'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-27

@description: TabNet multiclass tabular model.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import set_global_seed, to_numpy_x, to_numpy_y


@dataclass(frozen=True)
class TabNetConfig:
    n_d: int = 16
    n_a: int = 16
    n_steps: int = 4
    gamma: float = 1.5
    lambda_sparse: float = 1e-4
    learning_rate: float = 2e-2
    weight_decay: float = 1e-5
    max_epochs: int = 80
    patience: int = 10
    batch_size: int = 256
    virtual_batch_size: int = 64
    seed: int = 42
    verbose: int = 0


def make_model(cfg: TabNetConfig):
    try:
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError as e:
        raise ImportError(
            "pytorch-tabnet/torch no está instalado. Instala con: poetry add pytorch-tabnet torch"
        ) from e

    set_global_seed(cfg.seed, use_torch=True)

    model = TabNetClassifier(
        n_d=cfg.n_d,
        n_a=cfg.n_a,
        n_steps=cfg.n_steps,
        gamma=cfg.gamma,
        lambda_sparse=cfg.lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
        seed=cfg.seed,
        verbose=cfg.verbose,
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

    cfg = getattr(model, "_quant_risk_cfg", TabNetConfig())

    fit_kwargs = {
        "X_train": x_train,
        "y_train": y_train_np,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "batch_size": cfg.batch_size,
        "virtual_batch_size": cfg.virtual_batch_size,
        "num_workers": 0,
        "drop_last": False,
    }

    if X_valid is not None and y_valid is not None:
        x_valid = to_numpy_x(X_valid, dtype=np.float32)
        y_valid_np = to_numpy_y(y_valid)
        fit_kwargs["eval_set"] = [(x_valid, y_valid_np)]
        fit_kwargs["eval_name"] = ["valid"]
        fit_kwargs["eval_metric"] = ["accuracy"]

    model.fit(**fit_kwargs)
    return model


def predict(model: Any, X) -> np.ndarray:
    preds = model.predict(to_numpy_x(X, dtype=np.float32))
    return np.asarray(preds, dtype=int)


def predict_proba(model: Any, X) -> np.ndarray:
    proba = model.predict_proba(to_numpy_x(X, dtype=np.float32))
    return np.asarray(proba, dtype=float)
