'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-27

@description: Common utilities for tabular model training.
'''

from __future__ import annotations

import random
from typing import Any

import numpy as np


def set_global_seed(seed: int, use_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if not use_torch:
        return

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_numpy_x(X: Any, dtype=np.float32) -> np.ndarray:
    if hasattr(X, "to_numpy"):
        arr = X.to_numpy()
    else:
        arr = np.asarray(X)
    arr = np.asarray(arr, dtype=dtype)
    # Econometric chain models (EGARCH) can produce inf due to numerical instability.
    # XGBoost rejects inf; nan is treated as missing and handled internally.
    if np.any(np.isinf(arr)):
        arr = np.where(np.isinf(arr), np.nan, arr)
    return arr


def to_numpy_y(y: Any) -> np.ndarray:
    if hasattr(y, "to_numpy"):
        arr = y.to_numpy()
    else:
        arr = np.asarray(y)
    return np.asarray(arr, dtype=np.int64)


def build_variant_name(pooled: bool, use_sarimax: bool, use_garch: bool) -> str:
    base = "pooled" if pooled else "single"
    if use_sarimax and use_garch:
        feat = "sarimax_garch"
    elif use_sarimax:
        feat = "sarimax"
    elif use_garch:
        feat = "garch"
    else:
        feat = "tabular_only"
    return f"{base}_{feat}"
