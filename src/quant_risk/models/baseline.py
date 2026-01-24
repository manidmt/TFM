'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-24

@description: Module defining baseline (Logistic Regression and Random Forest) models for regime classification.
'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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