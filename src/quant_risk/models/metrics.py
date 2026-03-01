'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-24

@description: Module to compute classification metrics for regime prediction models.
'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report
)


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_recall: float
    class_recall: dict[int, float]
    report: str
    confusion: np.ndarray


def compute_metrics(y_true, y_pred, labels=(0, 1, 2)) -> Metrics:
    labels_list = list(labels)
    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    recalls = recall_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
    class_recall = {int(label): float(recalls[i]) for i, label in enumerate(labels_list)}
    macro_recall = float(np.mean(recalls)) if len(recalls) else float("nan")
    rep = classification_report(y_true, y_pred, labels=labels_list, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    return Metrics(
        accuracy=acc,
        macro_f1=macro,
        weighted_f1=weighted,
        macro_recall=macro_recall,
        class_recall=class_recall,
        report=rep,
        confusion=cm,
    )


def per_class_accuracy(cm: np.ndarray) -> dict[int, float]:
    out = {}
    for i in range(cm.shape[0]):
        denom = cm[i].sum()
        out[i] = float(cm[i, i] / denom) if denom else float("nan")
    return out
