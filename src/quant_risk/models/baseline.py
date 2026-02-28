'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-24

@description: Module defining baseline (Logistic Regression and Random Forest) models for regime classification.
'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class BaselineModels:
    logit: Pipeline
    rf: RandomForestClassifier


def make_models(random_state: int = 7) -> BaselineModels:
    logit = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    C=1.0,
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )

    return BaselineModels(logit=logit, rf=rf)


def majority_class(y_train) -> int:
    y_arr = np.asarray(y_train)
    if y_arr.size == 0:
        raise ValueError("Cannot compute majority class on empty training labels.")
    values, counts = np.unique(y_arr, return_counts=True)
    return int(values[np.argmax(counts)])


def predict_majority(n_samples: int, klass: int) -> np.ndarray:
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")
    return np.full(n_samples, int(klass), dtype=int)


def persistence_pred_from_regime(
    full_df: pd.DataFrame,
    horizon: int,
    regime_col: str = "regime",
) -> pd.Series:
    """
    Persistence baseline for y(t+h)=regime(t):
    for each row at date t, predict previous known regime at t-h.
    """
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    required = {"ticker", "date", regime_col}
    missing = required - set(full_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for persistence: {sorted(missing)}")

    tmp = full_df.sort_values(["ticker", "date"]).copy()
    pred = tmp.groupby("ticker")[regime_col].shift(int(horizon))
    pred.index = tmp.index
    return pred
