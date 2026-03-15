'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-28

@description: Walk-forward model selection for chained SARIMAX->GARCH->tabular
using robust delta macro F1 vs persistence baseline.
'''

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from quant_risk.datasets.make_dataset import DatasetConfig, make_dataset
from quant_risk.models.baseline import persistence_pred_from_regime
from quant_risk.models.metrics import compute_metrics
from quant_risk.models.tabular.gkg_change_detector import (
    GkgChangeDetectorConfig,
    build_design_matrix as build_gkg_change_design_matrix,
    fit as fit_gkg_change_detector,
    predict_proba as predict_proba_gkg_change_detector,
    select_gkg_feature_columns,
)
from quant_risk.models.tabular.tabpfn import (
    TabPFNConfig,
    fit as fit_tabpfn,
    make_model as make_tabpfn_model,
    predict_proba as predict_proba_tabpfn,
)
from quant_risk.models.tabular.xgb import (
    XGBConfig,
    fit as fit_xgb,
    make_model as make_xgb_model,
    predict_proba as predict_proba_xgb,
)
from quant_risk.tracking import mlflow_utils as tracking
from quant_risk.tracking.mlflow_utils import MlflowConfig

DEFAULT_CHAIN_VARIANTS_CONFIG = "src/quant_risk/models/econometric/chain_variants.yaml"
DEFAULT_TABPFN_VARIANTS_CONFIG = "src/quant_risk/models/tabular/tabpfn_variants.yaml"


def load_yaml(path: str) -> dict[str, Any]:
    """@brief Load a YAML file.

    @param path YAML file path.
    @return Parsed YAML content as dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    """@brief Snap a timestamp to month-end.

    @param ts Input timestamp.
    @return Month-end timestamp.
    """
    return pd.Timestamp(ts).to_period("M").to_timestamp("M")


def _normalize_order(value: Any) -> tuple[int, int, int]:
    """@brief Normalize a SARIMAX order to `(p, d, q)`.

    @param value Order as tuple/list or `p,d,q` string.
    @return Normalized `(p, d, q)` tuple.
    """
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) == 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
    raise ValueError(f"Invalid sarimax_order={value!r}; expected [p,d,q] or 'p,d,q'.")


def _normalize_exog_cols(value: Any) -> tuple[str, ...]:
    """@brief Normalize exogenous columns into a tuple.

    @param value Exogenous columns as tuple/list/string.
    @return Tuple of exogenous column names.
    """
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if str(v).strip())
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return ()
        if "|" in v:
            return tuple(tok.strip() for tok in v.split("|") if tok.strip())
        return (v,)
    raise ValueError(f"Invalid sarimax_chain_exog_cols={value!r}")


def _normalize_struct_cfg(row: dict[str, Any]) -> dict[str, Any]:
    """@brief Validate and normalize one structural chain configuration.

    @param row Raw structural config dictionary.
    @return Normalized structural config.
    """
    chain_mean_model = str(row.get("chain_mean_model", "sarimax")).lower()
    if chain_mean_model not in {"sarimax", "har"}:
        raise ValueError(
            f"Invalid chain_mean_model={chain_mean_model!r}. Use 'sarimax' or 'har'."
        )
    return {
        "chain_mean_model": chain_mean_model,
        "sarimax_order": _normalize_order(row.get("sarimax_order", (1, 0, 1))),
        "sarimax_chain_exog_cols": _normalize_exog_cols(
            row.get("sarimax_chain_exog_cols", ())
        ),
        "har_target_col": str(row.get("har_target_col", "rv_20")),
        "har_lag_1": int(row.get("har_lag_1", 1)),
        "har_lag_week": int(row.get("har_lag_week", 5)),
        "har_lag_month": int(row.get("har_lag_month", 22)),
        "har_exog_cols": _normalize_exog_cols(row.get("har_exog_cols", ())),
        "garch_p": int(row["garch_p"]),
        "garch_o": int(row.get("garch_o", 1)),
        "garch_q": int(row["garch_q"]),
        "garch_dist": str(row["garch_dist"]),
        "garch_vol": str(row.get("garch_vol", "Garch")),
        "garch_chain_agg": str(row["garch_chain_agg"]),
        "garch_scale": float(row["garch_scale"]),
    }


def _load_chain_variants_from_yaml(profile: str, variants_path: str) -> list[dict[str, Any]]:
    """@brief Load chain structural variants from YAML.

    @param profile Grid profile (`promising` or `full`).
    @param variants_path YAML path with variant definitions.
    @return List of normalized structural configs.
    """
    if not Path(variants_path).exists():
        return []
    doc = load_yaml(variants_path)
    profiles = doc.get("profiles", {}) if isinstance(doc, dict) else {}
    if profile == "promising":
        rows = profiles.get("promising", [])
        if not isinstance(rows, list):
            raise ValueError(
                f"{variants_path}: profiles.promising must be a list of structural configs."
            )
        return [_normalize_struct_cfg(dict(r)) for r in rows]

    axes = profiles.get("full_axes", {})
    if not isinstance(axes, dict):
        raise ValueError(f"{variants_path}: profiles.full_axes must be a mapping.")
    mean_models = [str(v).lower() for v in axes.get("chain_mean_model", ["sarimax"])]
    valid_mean_models = {"sarimax", "har"}
    invalid_mean_models = sorted({m for m in mean_models if m not in valid_mean_models})
    if invalid_mean_models:
        raise ValueError(
            f"{variants_path}: profiles.full_axes.chain_mean_model contains invalid "
            f"entries {invalid_mean_models!r}; allowed values are "
            f"{sorted(valid_mean_models)!r}."
        )
    orders = [_normalize_order(v) for v in axes.get("sarimax_order", [(1, 0, 1)])]
    exogs = [_normalize_exog_cols(v) for v in axes.get("sarimax_chain_exog_cols", [()])]
    har_targets = [str(v) for v in axes.get("har_target_col", ["rv_20"])]
    har_lag_1_vals = [int(v) for v in axes.get("har_lag_1", [1])]
    har_lag_week_vals = [int(v) for v in axes.get("har_lag_week", [5])]
    har_lag_month_vals = [int(v) for v in axes.get("har_lag_month", [22])]
    har_exogs = [_normalize_exog_cols(v) for v in axes.get("har_exog_cols", [()])]
    p_vals = [int(v) for v in axes.get("garch_p", [])]
    o_vals = [int(v) for v in axes.get("garch_o", [1])]
    q_vals = [int(v) for v in axes.get("garch_q", [])]
    dists = [str(v) for v in axes.get("garch_dist", [])]
    vols = [str(v) for v in axes.get("garch_vol", ["Garch"])]
    aggs = [str(v) for v in axes.get("garch_chain_agg", [])]
    scales = [float(v) for v in axes.get("garch_scale", [])]
    if not all(
        [
            mean_models,
            orders,
            exogs,
            har_targets,
            har_lag_1_vals,
            har_lag_week_vals,
            har_lag_month_vals,
            har_exogs,
            p_vals,
            o_vals,
            q_vals,
            dists,
            vols,
            aggs,
            scales,
        ]
    ):
        raise ValueError(
            f"{variants_path}: profiles.full_axes is missing required non-empty keys."
        )
    out: list[dict[str, Any]] = []
    for mean_model, order, exog, har_target, har_l1, har_lw, har_lm, har_exog, p, o, q, dist, vol, agg, scale in itertools.product(
        mean_models,
        orders,
        exogs,
        har_targets,
        har_lag_1_vals,
        har_lag_week_vals,
        har_lag_month_vals,
        har_exogs,
        p_vals,
        o_vals,
        q_vals,
        dists,
        vols,
        aggs,
        scales,
    ):
        out.append(
            {
                "chain_mean_model": str(mean_model),
                "sarimax_order": order,
                "sarimax_chain_exog_cols": exog,
                "har_target_col": str(har_target),
                "har_lag_1": int(har_l1),
                "har_lag_week": int(har_lw),
                "har_lag_month": int(har_lm),
                "har_exog_cols": har_exog,
                "garch_p": int(p),
                "garch_o": int(o),
                "garch_q": int(q),
                "garch_dist": str(dist),
                "garch_vol": str(vol),
                "garch_chain_agg": str(agg),
                "garch_scale": float(scale),
            }
        )
    return out


def build_folds(
    min_train_end: str,
    max_valid_end: str,
    valid_months: int,
    step_months: int,
) -> list[tuple[str, str]]:
    """@brief Build walk-forward fold boundaries.

    @param min_train_end First train end date.
    @param max_valid_end Last validation end date.
    @param valid_months Validation window size in months.
    @param step_months Fold step size in months.
    @return List of `(train_end, valid_end)` tuples.
    """
    folds: list[tuple[str, str]] = []
    train_end = month_end(pd.to_datetime(min_train_end))
    max_valid = month_end(pd.to_datetime(max_valid_end))

    while True:
        valid_end = month_end(train_end + pd.DateOffset(months=valid_months))
        if valid_end > max_valid:
            break
        folds.append((str(train_end.date()), str(valid_end.date())))
        train_end = month_end(train_end + pd.DateOffset(months=step_months))

    return folds


def make_structural_configs(profile: str, variants_path: str = DEFAULT_CHAIN_VARIANTS_CONFIG) -> list[dict[str, Any]]:
    """@brief Build structural chain configuration grid.

    @param profile Grid profile (`promising` or `full`).
    @param variants_path YAML path for structural variants.
    @return List of structural configurations.
    """
    yaml_cfgs = _load_chain_variants_from_yaml(profile, variants_path)
    if yaml_cfgs:
        return yaml_cfgs
    if profile == "promising":
        return [
            {
                "chain_mean_model": "har",
                "sarimax_order": (1, 0, 1),
                "sarimax_chain_exog_cols": (),
                "har_target_col": "rv_20",
                "har_lag_1": 1,
                "har_lag_week": 5,
                "har_lag_month": 22,
                "har_exog_cols": (),
                "garch_p": 2,
                "garch_o": 1,
                "garch_q": 2,
                "garch_dist": "tstudent",
                "garch_vol": "Garch",
                "garch_chain_agg": "rms",
                "garch_scale": 100.0,
            },
            {
                "chain_mean_model": "sarimax",
                "sarimax_order": (2, 0, 1),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "har_target_col": "rv_20",
                "har_lag_1": 1,
                "har_lag_week": 5,
                "har_lag_month": 22,
                "har_exog_cols": (),
                "garch_p": 2,
                "garch_o": 1,
                "garch_q": 2,
                "garch_dist": "skewt",
                "garch_vol": "Garch",
                "garch_chain_agg": "rms",
                "garch_scale": 100.0,
            },
            {
                "chain_mean_model": "sarimax",
                "sarimax_order": (2, 0, 1),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "har_target_col": "rv_20",
                "har_lag_1": 1,
                "har_lag_week": 5,
                "har_lag_month": 22,
                "har_exog_cols": (),
                "garch_p": 2,
                "garch_o": 1,
                "garch_q": 2,
                "garch_dist": "tstudent",
                "garch_vol": "Garch",
                "garch_chain_agg": "rms",
                "garch_scale": 100.0,
            },
            {
                "chain_mean_model": "sarimax",
                "sarimax_order": (2, 0, 1),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "har_target_col": "rv_20",
                "har_lag_1": 1,
                "har_lag_week": 5,
                "har_lag_month": 22,
                "har_exog_cols": (),
                "garch_p": 1,
                "garch_o": 1,
                "garch_q": 2,
                "garch_dist": "tstudent",
                "garch_vol": "Garch",
                "garch_chain_agg": "mean",
                "garch_scale": 80.0,
            },
            {
                "chain_mean_model": "sarimax",
                "sarimax_order": (1, 0, 1),
                "sarimax_chain_exog_cols": (),
                "har_target_col": "rv_20",
                "har_lag_1": 1,
                "har_lag_week": 5,
                "har_lag_month": 22,
                "har_exog_cols": (),
                "garch_p": 1,
                "garch_o": 1,
                "garch_q": 1,
                "garch_dist": "tstudent",
                "garch_vol": "Garch",
                "garch_chain_agg": "rms",
                "garch_scale": 100.0,
            },
            {
                "chain_mean_model": "sarimax",
                "sarimax_order": (1, 0, 0),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "har_target_col": "rv_20",
                "har_lag_1": 1,
                "har_lag_week": 5,
                "har_lag_month": 22,
                "har_exog_cols": (),
                "garch_p": 1,
                "garch_o": 1,
                "garch_q": 1,
                "garch_dist": "tstudent",
                "garch_vol": "Garch",
                "garch_chain_agg": "last",
                "garch_scale": 80.0,
            },
            {
                "chain_mean_model": "sarimax",
                "sarimax_order": (2, 0, 1),
                "sarimax_chain_exog_cols": (),
                "har_target_col": "rv_20",
                "har_lag_1": 1,
                "har_lag_week": 5,
                "har_lag_month": 22,
                "har_exog_cols": (),
                "garch_p": 2,
                "garch_o": 1,
                "garch_q": 1,
                "garch_dist": "tstudent",
                "garch_vol": "Garch",
                "garch_chain_agg": "rms",
                "garch_scale": 100.0,
            },
            {
                "chain_mean_model": "sarimax",
                "sarimax_order": (1, 0, 1),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "har_target_col": "rv_20",
                "har_lag_1": 1,
                "har_lag_week": 5,
                "har_lag_month": 22,
                "har_exog_cols": (),
                "garch_p": 2,
                "garch_o": 1,
                "garch_q": 1,
                "garch_dist": "tstudent",
                "garch_vol": "Garch",
                "garch_chain_agg": "mean",
                "garch_scale": 100.0,
            },
        ]

    mean_models = ["sarimax", "har"]
    orders = [(1, 0, 1), (2, 0, 1)]
    exogs = [(), ("net_liquidity_diff",)]
    har_targets = ["rv_20"]
    har_lag_1_vals = [1]
    har_lag_week_vals = [5]
    har_lag_month_vals = [22]
    har_exogs = [()]
    p_vals = [1, 2]
    o_vals = [1]
    q_vals = [1, 2]
    dists = ["tstudent", "skewt"]
    vols = ["Garch"]
    aggs = ["rms", "mean"]
    scales = [80.0, 100.0]

    out = []
    for mean_model, order, exog, har_target, har_l1, har_lw, har_lm, har_exog, p, o, q, dist, vol, agg, scale in itertools.product(
        mean_models,
        orders,
        exogs,
        har_targets,
        har_lag_1_vals,
        har_lag_week_vals,
        har_lag_month_vals,
        har_exogs,
        p_vals,
        o_vals,
        q_vals,
        dists,
        vols,
        aggs,
        scales,
    ):
        out.append(
            {
                "chain_mean_model": str(mean_model),
                "sarimax_order": order,
                "sarimax_chain_exog_cols": exog,
                "har_target_col": str(har_target),
                "har_lag_1": int(har_l1),
                "har_lag_week": int(har_lw),
                "har_lag_month": int(har_lm),
                "har_exog_cols": har_exog,
                "garch_p": int(p),
                "garch_o": int(o),
                "garch_q": int(q),
                "garch_dist": str(dist),
                "garch_vol": str(vol),
                "garch_chain_agg": str(agg),
                "garch_scale": float(scale),
            }
        )
    return out


def make_xgb_configs(profile: str) -> list[dict[str, Any]]:
    """@brief Build XGBoost hyperparameter grid.

    @param profile Grid profile (`promising` or `full`).
    @return List of XGBoost model configs.
    """
    if profile == "promising":
        return [
            {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 500,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.95,
                "colsample_bytree": 0.9,
                "min_child_weight": 1.0,
                "reg_lambda": 1.5,
            },
        ]

    n_estimators = [300, 500]
    max_depth = [4, 5]
    learning_rate = [0.05]
    subsample = [0.9]
    colsample_bytree = [0.9]
    min_child_weight = [1.0]
    reg_lambda = [1.0]

    out = []
    for ne, md, lr, ss, cs, mcw, rlam in itertools.product(
        n_estimators,
        max_depth,
        learning_rate,
        subsample,
        colsample_bytree,
        min_child_weight,
        reg_lambda,
    ):
        out.append(
            {
                "n_estimators": int(ne),
                "max_depth": int(md),
                "learning_rate": float(lr),
                "subsample": float(ss),
                "colsample_bytree": float(cs),
                "min_child_weight": float(mcw),
                "reg_lambda": float(rlam),
            }
        )
    return out


def make_tabpfn_configs(
    profile: str,
    *,
    device: str = "auto",
    n_preprocessing_jobs: int = 1,
    model_version: str = "v2",
    variants_path: str = DEFAULT_TABPFN_VARIANTS_CONFIG,
) -> list[dict[str, Any]]:
    """@brief Build TabPFN hyperparameter grid.

    @param profile Grid profile (`promising` or `full`).
    @param device Execution device for TabPFN.
    @param n_preprocessing_jobs Number of preprocessing jobs.
    @param model_version TabPFN checkpoint family (`v2` or `v2.5`).
    @param variants_path YAML path with TabPFN variants.
    @return List of TabPFN model configs.
    """
    model_version_l = str(model_version).lower()
    if model_version_l not in {"v2", "v2.5"}:
        raise ValueError(f"Invalid tabpfn model_version={model_version!r}. Use 'v2' or 'v2.5'.")
    model_path = (
        "tabpfn-v2-classifier-v2_default.ckpt"
        if model_version_l == "v2"
        else "tabpfn-v2.5-classifier-v2.5_default.ckpt"
    )
    if Path(variants_path).exists():
        doc = load_yaml(variants_path)
        profiles = doc.get("profiles", {}) if isinstance(doc, dict) else {}
        rows = profiles.get(profile, [])
        if not isinstance(rows, list):
            raise ValueError(
                f"{variants_path}: profiles.{profile} must be a list of TabPFN configs."
            )
        if rows:
            out: list[dict[str, Any]] = []
            for row in rows:
                rr = dict(row)
                rr["model_path"] = str(rr.get("model_path", model_path))
                rr["device"] = str(rr.get("device", device))
                rr["n_preprocessing_jobs"] = int(
                    rr.get("n_preprocessing_jobs", n_preprocessing_jobs)
                )
                rr["fit_mode"] = str(rr.get("fit_mode", "fit_preprocessors"))
                rr["memory_saving_mode"] = str(rr.get("memory_saving_mode", "auto"))
                rr["ignore_pretraining_limits"] = bool(
                    rr.get("ignore_pretraining_limits", True)
                )
                rr["inference_precision"] = str(rr.get("inference_precision", "auto"))
                rr["random_state"] = int(rr.get("random_state", 42))
                rr["n_estimators"] = int(rr.get("n_estimators", 1))
                rr["softmax_temperature"] = float(rr.get("softmax_temperature", 0.9))
                rr["balance_probabilities"] = bool(rr.get("balance_probabilities", False))
                rr["average_before_softmax"] = bool(rr.get("average_before_softmax", False))
                out.append(rr)
            return out

    base = {
        "device": str(device),
        "n_preprocessing_jobs": int(n_preprocessing_jobs),
        "fit_mode": "fit_preprocessors",
        "memory_saving_mode": "auto",
        "ignore_pretraining_limits": True,
        "inference_precision": "auto",
        "model_path": model_path,
        "random_state": 42,
    }
    if profile == "promising":
        return [
            {
                **base,
                "n_estimators": 1,
                "softmax_temperature": 0.9,
                "balance_probabilities": False,
                "average_before_softmax": False,
            },
            {
                **base,
                "n_estimators": 4,
                "softmax_temperature": 0.9,
                "balance_probabilities": False,
                "average_before_softmax": False,
            },
        ]

    return [
        {
            **base,
            "n_estimators": 1,
            "softmax_temperature": 0.9,
            "balance_probabilities": False,
            "average_before_softmax": False,
        },
        {
            **base,
            "n_estimators": 4,
            "softmax_temperature": 0.9,
            "balance_probabilities": False,
            "average_before_softmax": False,
        },
        {
            **base,
            "n_estimators": 8,
            "softmax_temperature": 0.8,
            "balance_probabilities": False,
            "average_before_softmax": False,
        },
    ]


def _high_vol_recall(metrics_obj: Any) -> float:
    """@brief Get recall for the highest-volatility class.

    @param metrics_obj Metrics-like object with `class_recall`.
    @return Recall for highest class label, or NaN.
    """
    if not getattr(metrics_obj, "class_recall", None):
        return float("nan")
    high_vol_label = max(metrics_obj.class_recall.keys())
    return float(metrics_obj.class_recall.get(high_vol_label, float("nan")))


def _one_hot_probs(y: np.ndarray, n_classes: int) -> np.ndarray:
    """@brief Convert class labels to one-hot probability rows.

    @param y Integer class labels.
    @param n_classes Number of classes.
    @return One-hot matrix of shape `(n_samples, n_classes)`.
    """
    y_arr = np.asarray(y, dtype=int)
    out = np.zeros((len(y_arr), int(n_classes)), dtype=float)
    if len(y_arr):
        out[np.arange(len(y_arr)), y_arr] = 1.0
    return out


def _parse_unit_interval_list(spec: str, *, name: str) -> tuple[float, ...]:
    """@brief Parse comma-separated values constrained to `[0, 1]`.

    @param spec Comma-separated numeric string.
    @param name Parameter name for error messages.
    @return Sorted unique tuple of floats.
    """
    vals: list[float] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"{name} value must be in [0,1], got {v}")
        vals.append(v)
    if not vals:
        raise ValueError(f"{name} is empty")
    return tuple(sorted(set(vals)))


def _parse_blend_alphas(spec: str) -> tuple[float, ...]:
    """@brief Parse blend alpha candidates.

    @param spec Comma-separated alpha values.
    @return Valid alpha tuple.
    """
    return _parse_unit_interval_list(spec, name="blend_alphas")


def _parse_blend_conf_betas(spec: str) -> tuple[float, ...]:
    """@brief Parse confidence-beta candidates for blending.

    @param spec Comma-separated beta values.
    @return Valid beta tuple.
    """
    return _parse_unit_interval_list(spec, name="blend_conf_betas")


def _parse_gate_thresholds(spec: str) -> tuple[float, ...]:
    """@brief Parse gating confidence thresholds.

    @param spec Comma-separated threshold values.
    @return Valid threshold tuple.
    """
    return _parse_unit_interval_list(spec, name="gate_thresholds")


def _parse_gkg_change_gate_thresholds(spec: str) -> tuple[float, ...]:
    """@brief Parse sidecar p_change thresholds used to gate persistence vs chain."""
    return _parse_unit_interval_list(spec, name="gkg_change_gate_thresholds")


def _parse_gkg_change_alpha_weights(spec: str) -> tuple[float, ...]:
    """@brief Parse sidecar p_change weights used to attenuate alpha."""
    return _parse_unit_interval_list(spec, name="gkg_change_alpha_weights")


def _parse_gkg_change_alpha_weights_by_asset(spec: str) -> dict[str, tuple[float, ...]]:
    """@brief Parse asset-specific alpha-weight grids.

    Syntax example:
        `BTC-USD=0.0,0.1,0.2;TLT=0.0,0.05,0.1`

    @param spec Semicolon-separated `asset=weights` assignments.
    @return Mapping `{asset: weight_grid}`.
    """
    txt = str(spec).strip()
    if not txt:
        return {}
    out: dict[str, tuple[float, ...]] = {}
    for block in txt.split(";"):
        block = block.strip()
        if not block:
            continue
        if "=" not in block:
            raise ValueError(
                "gkg_change_alpha_weights_by_asset entries must use `asset=weights` syntax, "
                f"got {block!r}"
            )
        asset_name, weight_spec = block.split("=", 1)
        asset_key = str(asset_name).strip()
        if not asset_key:
            raise ValueError(
                "gkg_change_alpha_weights_by_asset contains an empty asset key."
            )
        out[asset_key] = _parse_unit_interval_list(
            weight_spec,
            name=f"gkg_change_alpha_weights_by_asset[{asset_key}]",
        )
    return out


def _resolve_asset_specific_grid(
    *,
    default_grid: tuple[float, ...],
    asset_overrides: dict[str, tuple[float, ...]],
    asset_name: str,
    tickers: tuple[str, ...],
) -> tuple[float, ...]:
    """@brief Resolve an optional asset-specific grid override.

    Resolution order:
    1. `asset_name`
    2. first matching ticker alias in `tickers`
    3. `default_grid`

    @param default_grid Fallback grid.
    @param asset_overrides Optional asset-specific overrides.
    @param asset_name Current asset/group identifier.
    @param tickers Tickers included in current asset/group.
    @return Selected grid for this asset.
    """
    for key in (str(asset_name), *tuple(str(t) for t in tickers)):
        if key in asset_overrides:
            return tuple(asset_overrides[key])
    return tuple(default_grid)


def _parse_class_threshold_grid(
    spec: str,
    *,
    n_classes: int = 3,
) -> tuple[tuple[float, ...], ...]:
    """@brief Parse class-threshold vectors from CLI syntax.

    @param spec Semicolon-separated vectors, each comma-separated.
    @param n_classes Expected values per vector.
    @return Tuple of class-threshold vectors.
    """
    vectors: list[tuple[float, ...]] = []
    text = str(spec).strip()
    if not text:
        return ((1.0,) * n_classes,)
    for vec_txt in text.split(";"):
        vec_txt = vec_txt.strip()
        if not vec_txt:
            continue
        parts = [p.strip() for p in vec_txt.split(",") if p.strip()]
        if len(parts) != n_classes:
            raise ValueError(
                f"class_threshold_grid vector must have {n_classes} values, got {parts!r}"
            )
        vec = tuple(float(v) for v in parts)
        if any(v <= 0.0 for v in vec):
            raise ValueError(f"class_threshold_grid values must be > 0, got {vec!r}")
        vectors.append(vec)
    if not vectors:
        return ((1.0,) * n_classes,)
    dedup = []
    seen = set()
    for v in vectors:
        if v not in seen:
            seen.add(v)
            dedup.append(v)
    return tuple(dedup)


def _format_class_thresholds(v: np.ndarray | tuple[float, ...] | list[float]) -> str:
    """@brief Serialize class thresholds to compact string form.

    @param v Class-threshold vector.
    @return Thresholds joined with `|`.
    """
    arr = np.asarray(v, dtype=float).reshape(-1)
    return "|".join(f"{x:.6g}" for x in arr)


def _parse_csv_list(spec: Any) -> tuple[str, ...]:
    """@brief Parse a comma-separated token list preserving order and uniqueness."""
    if spec is None:
        return ()
    toks = [str(t).strip() for t in str(spec).split(",") if str(t).strip()]
    out: list[str] = []
    seen: set[str] = set()
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return tuple(out)


def _parse_class_thresholds(value: Any, *, n_classes: int = 3) -> np.ndarray:
    """@brief Normalize class thresholds to numpy array.

    @param value Threshold input as string/list/array.
    @param n_classes Expected number of classes.
    @return Array of shape `(n_classes,)`.
    """
    if isinstance(value, str):
        if not value.strip():
            return np.ones(n_classes, dtype=float)
        toks = [t.strip() for t in value.replace("|", ",").split(",") if t.strip()]
        arr = np.asarray([float(t) for t in toks], dtype=float)
    else:
        arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (n_classes,):
        raise ValueError(
            f"class_thresholds must have shape ({n_classes},), got {arr.shape}"
        )
    if np.any(arr <= 0.0):
        raise ValueError(f"class_thresholds must be > 0, got {arr}")
    return arr.astype(float)


def _normalize_proba_rows(proba: np.ndarray) -> np.ndarray:
    """@brief Sanitize and row-normalize probability matrix.

    @param proba Probability matrix.
    @return Valid probability matrix with row sums equal to 1.
    """
    p = np.asarray(proba, dtype=float)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)
    if p.ndim != 2 or p.shape[1] <= 0:
        raise ValueError(f"Invalid probability array shape: {p.shape}")
    row_sum = np.sum(p, axis=1, keepdims=True)
    bad = (~np.isfinite(row_sum)) | (row_sum <= 0.0)
    if np.any(bad):
        p = p.copy()
        p[bad.reshape(-1), :] = 1.0 / float(p.shape[1])
        row_sum = np.sum(p, axis=1, keepdims=True)
    return p / row_sum


def _fit_probability_calibrator(
    proba: np.ndarray,
    y_true: np.ndarray,
    method: str,
) -> dict[str, Any]:
    """@brief Fit a one-vs-rest probability calibrator.

    @param proba Base probabilities.
    @param y_true Ground-truth labels.
    @param method Calibration method (`none`, `platt`, `isotonic`).
    @return Calibrator payload (method + fitted per-class models).
    """
    method_l = str(method).lower().strip()
    if method_l in {"", "none"}:
        return {"method": "none", "models": [], "n_classes": int(np.asarray(proba).shape[1])}

    if method_l not in {"platt", "isotonic"}:
        raise ValueError(
            f"Invalid calibration method={method!r}. Use one of: none, platt, isotonic."
        )

    p = _normalize_proba_rows(proba)
    y = np.asarray(y_true, dtype=int).reshape(-1)
    if p.shape[0] != y.shape[0]:
        raise ValueError(
            f"Calibration inputs have different lengths: proba={p.shape[0]} y={y.shape[0]}"
        )

    n_classes = int(p.shape[1])
    models: list[Any | None] = []
    for cls in range(n_classes):
        y_bin = (y == cls).astype(int)
        if int(np.unique(y_bin).size) < 2:
            models.append(None)
            continue
        x_col = p[:, cls]
        if method_l == "platt":
            clf = LogisticRegression(max_iter=300, solver="lbfgs", random_state=42)
            clf.fit(x_col.reshape(-1, 1), y_bin)
            models.append(clf)
        else:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(x_col, y_bin)
            models.append(ir)

    return {"method": method_l, "models": models, "n_classes": n_classes}


def _apply_probability_calibrator(
    proba: np.ndarray,
    calibrator: dict[str, Any] | None,
) -> np.ndarray:
    """@brief Apply calibrated probabilities and re-normalize rows.

    @param proba Input probabilities.
    @param calibrator Calibrator object from `_fit_probability_calibrator`.
    @return Calibrated probability matrix.
    """
    if not calibrator or str(calibrator.get("method", "none")) == "none":
        return _normalize_proba_rows(proba)

    method = str(calibrator.get("method", "none"))
    models = calibrator.get("models", [])
    p = _normalize_proba_rows(proba)
    out = np.zeros_like(p, dtype=float)
    for cls in range(p.shape[1]):
        model = models[cls] if cls < len(models) else None
        if model is None:
            out[:, cls] = p[:, cls]
            continue
        x_col = p[:, cls]
        if method == "platt":
            out[:, cls] = np.asarray(model.predict_proba(x_col.reshape(-1, 1))[:, 1], dtype=float)
        elif method == "isotonic":
            out[:, cls] = np.asarray(model.predict(x_col), dtype=float)
        else:
            out[:, cls] = p[:, cls]
    return _normalize_proba_rows(out)


def _predict_with_class_thresholds(
    proba: np.ndarray,
    class_thresholds: np.ndarray | tuple[float, ...] | list[float] | None = None,
) -> np.ndarray:
    """@brief Predict labels after class-wise threshold reweighting.

    @param proba Probability matrix.
    @param class_thresholds Optional class-wise divisors.
    @return Predicted class labels.
    """
    p = _normalize_proba_rows(proba)
    n_classes = int(p.shape[1])
    thr = (
        np.ones(n_classes, dtype=float)
        if class_thresholds is None
        else _parse_class_thresholds(class_thresholds, n_classes=n_classes)
    )
    scores = p / thr.reshape(1, -1)
    return np.argmax(scores, axis=1).astype(int)


def _subset_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """@brief Compute macro-F1 on a masked subset.

    @param y_true Ground-truth labels.
    @param y_pred Predicted labels.
    @param mask Boolean mask selecting the subset.
    @return Macro-F1 on subset or NaN when empty.
    """
    m = np.asarray(mask, dtype=bool)
    if int(np.sum(m)) == 0:
        return float("nan")
    mm = compute_metrics(np.asarray(y_true, dtype=int)[m], np.asarray(y_pred, dtype=int)[m])
    return float(mm.macro_f1)


def _safe_binary_auc(y_true_bin: np.ndarray, p_pos: np.ndarray) -> float:
    """@brief Compute ROC-AUC for binary labels with safe fallbacks."""
    y = np.asarray(y_true_bin, dtype=int).reshape(-1)
    p = np.asarray(p_pos, dtype=float).reshape(-1)
    if y.size == 0 or p.size == 0 or y.size != p.size:
        return float("nan")
    if int(np.unique(y).size) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def _safe_binary_brier(y_true_bin: np.ndarray, p_pos: np.ndarray) -> float:
    """@brief Compute Brier score for binary labels with safe fallbacks."""
    y = np.asarray(y_true_bin, dtype=int).reshape(-1)
    p = np.asarray(p_pos, dtype=float).reshape(-1)
    if y.size == 0 or p.size == 0 or y.size != p.size:
        return float("nan")
    try:
        return float(brier_score_loss(y, p))
    except Exception:
        return float("nan")


def _safe_nanmean(arr: np.ndarray | Sequence[float]) -> float:
    """@brief Mean over finite values only, returning NaN when none exist."""
    x = np.asarray(arr, dtype=float).reshape(-1)
    if x.size == 0:
        return float("nan")
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _pick_sigma_t_from_matrix(
    X: np.ndarray,
    feature_cols: Sequence[str],
) -> np.ndarray | None:
    """@brief Extract a volatility context signal from the full feature matrix when present."""
    cols = [str(c) for c in feature_cols]
    candidates = ("sigma_t", "garch_sigma_t", "egarch_sigma_t", "gjrgarch_sigma_t")
    idx = next((cols.index(c) for c in candidates if c in cols), None)
    if idx is None:
        return None
    return np.asarray(X[:, idx], dtype=float)


def _fit_eval_gkg_change_detector(
    *,
    feature_cols: Sequence[str],
    x_train_aligned: np.ndarray,
    y_train_target_aligned: np.ndarray,
    regime_prev_train: np.ndarray,
    persist_train_pred: np.ndarray,
    x_eval_aligned: np.ndarray,
    y_eval_target_aligned: np.ndarray,
    regime_prev_eval: np.ndarray,
    persist_eval_pred: np.ndarray,
    model_type: str,
    calibration_method: str,
    context_cols: tuple[str, ...],
    xgb_n_jobs: int,
) -> dict[str, Any]:
    """@brief Train/evaluate a GKG-only binary change detector with calibrated probabilities."""
    y_train_change = (
        np.asarray(y_train_target_aligned, dtype=int) != np.asarray(persist_train_pred, dtype=int)
    ).astype(int)
    y_eval_change = (
        np.asarray(y_eval_target_aligned, dtype=int) != np.asarray(persist_eval_pred, dtype=int)
    ).astype(int)

    gkg_cols = select_gkg_feature_columns(feature_cols)
    sigma_train = _pick_sigma_t_from_matrix(x_train_aligned, feature_cols)
    sigma_eval = _pick_sigma_t_from_matrix(x_eval_aligned, feature_cols)
    context_cols_eff = tuple(str(c) for c in context_cols)
    if "sigma_t" in context_cols_eff and (sigma_train is None or sigma_eval is None):
        context_cols_eff = tuple(c for c in context_cols_eff if c != "sigma_t")

    X_train_change, used_feature_names = build_gkg_change_design_matrix(
        X_all=np.asarray(x_train_aligned, dtype=np.float32),
        feature_cols=feature_cols,
        gkg_feature_cols=gkg_cols,
        context_cols=context_cols_eff,
        regime_prev=np.asarray(regime_prev_train, dtype=float),
        sigma_t=sigma_train,
    )
    X_eval_change, _ = build_gkg_change_design_matrix(
        X_all=np.asarray(x_eval_aligned, dtype=np.float32),
        feature_cols=feature_cols,
        gkg_feature_cols=gkg_cols,
        context_cols=context_cols_eff,
        regime_prev=np.asarray(regime_prev_eval, dtype=float),
        sigma_t=sigma_eval,
    )

    if X_train_change.shape[1] == 0:
        return {
            "enabled": False,
            "reason": "no_gkg_features",
            "gkg_change_model": str(model_type).lower(),
            "gkg_change_context_cols": ",".join(context_cols_eff),
            "gkg_change_n_features": 0,
            "gkg_change_feature_names": "",
        }
    if int(np.unique(y_train_change).size) < 2:
        return {
            "enabled": False,
            "reason": "degenerate_train_target",
            "gkg_change_model": str(model_type).lower(),
            "gkg_change_context_cols": ",".join(context_cols_eff),
            "gkg_change_n_features": int(X_train_change.shape[1]),
            "gkg_change_feature_names": "|".join(used_feature_names),
            "gkg_change_train_change_rate": float(np.mean(y_train_change)),
        }

    cfg = GkgChangeDetectorConfig(
        model_type=str(model_type).lower(),
        calibration_method=str(calibration_method).lower(),
        random_state=42,
        seed=42,
        xgb_n_jobs=int(xgb_n_jobs),
    )
    artifacts = fit_gkg_change_detector(
        cfg,
        X_train_change,
        y_train_change,
        feature_names=tuple(used_feature_names),
    )
    proba_eval_change = predict_proba_gkg_change_detector(artifacts, X_eval_change)
    p_change_eval = np.asarray(proba_eval_change[:, 1], dtype=float)
    pred_change_eval = (p_change_eval >= 0.5).astype(int)
    m = compute_metrics(y_eval_change, pred_change_eval)

    return {
        "enabled": True,
        "reason": "ok",
        "gkg_change_model": str(model_type).lower(),
        "gkg_change_context_cols": ",".join(context_cols_eff),
        "gkg_change_n_features": int(X_train_change.shape[1]),
        "gkg_change_feature_names": "|".join(used_feature_names),
        "gkg_change_train_change_rate": float(np.mean(y_train_change)),
        "gkg_change_eval_change_rate": float(np.mean(y_eval_change)),
        "gkg_change_eval_acc": float(m.accuracy),
        "gkg_change_eval_macro_f1": float(m.macro_f1),
        "gkg_change_eval_weighted_f1": float(m.weighted_f1),
        "gkg_change_eval_macro_recall": float(m.macro_recall),
        "gkg_change_eval_auc": _safe_binary_auc(y_eval_change, p_change_eval),
        "gkg_change_eval_brier": _safe_binary_brier(y_eval_change, p_change_eval),
        "gkg_change_eval_p_change_mean": float(np.mean(p_change_eval)),
        "gkg_change_eval_p_change_std": float(np.std(p_change_eval)),
        "gkg_change_eval_pred_change_rate": float(np.mean(pred_change_eval)),
        "gkg_change_eval_p_no_change_mean": float(np.mean(1.0 - p_change_eval)),
        "gkg_change_calibration_method": str(calibration_method).lower(),
        "gkg_change_p_change_eval": p_change_eval.astype(float),
        "gkg_change_y_eval": y_eval_change.astype(int),
        "gkg_change_pred_eval": pred_change_eval.astype(int),
    }


def _blend_strategy_predict(
    proba_chain: np.ndarray,
    y_persist: np.ndarray,
    alpha: float,
    beta: float = 0.0,
    class_thresholds: np.ndarray | tuple[float, ...] | list[float] | None = None,
    gate_threshold: float = 0.0,
    p_change: np.ndarray | None = None,
    change_gate_threshold: float = 0.0,
    change_alpha_weight: float = 0.0,
) -> dict[str, Any]:
    """@brief Predict with blend + confidence gating against persistence.

    @param proba_chain Chain model probabilities.
    @param y_persist Persistence labels.
    @param alpha Base chain blend weight.
    @param beta Confidence-adaptive blend weight factor.
    @param class_thresholds Optional class-wise thresholds.
    @param gate_threshold Confidence threshold to route to persistence.
    @param p_change Optional sidecar probability of regime change.
    @param change_gate_threshold Threshold below which persistence is forced.
    @param change_alpha_weight Weight used to attenuate alpha as a function of p_change.
    @return Prediction payload with labels, probabilities, and diagnostics.
    """
    p_chain = _normalize_proba_rows(proba_chain)
    alpha_c = float(np.clip(alpha, 0.0, 1.0))
    beta_c = float(np.clip(beta, 0.0, 1.0))
    gate_t = float(np.clip(gate_threshold, 0.0, 1.0))
    persist_proba = _one_hot_probs(y_persist, n_classes=int(p_chain.shape[1]))

    if beta_c <= 0.0:
        alpha_vec_base = np.full(len(p_chain), alpha_c, dtype=float)
    else:
        conf_chain = np.max(p_chain, axis=1).astype(float)
        alpha_vec_base = np.clip((1.0 - beta_c) * alpha_c + beta_c * conf_chain, 0.0, 1.0)

    if p_change is None:
        p_change_vec = np.full(len(p_chain), np.nan, dtype=float)
    else:
        p_change_vec = np.asarray(p_change, dtype=float).reshape(-1)
        if p_change_vec.shape[0] != p_chain.shape[0]:
            raise ValueError(
                "p_change length must match proba_chain rows: "
                f"{p_change_vec.shape[0]} vs {p_chain.shape[0]}"
            )
        p_change_vec = np.nan_to_num(p_change_vec, nan=1.0, posinf=1.0, neginf=0.0)
        p_change_vec = np.clip(p_change_vec, 0.0, 1.0)

    change_alpha_w = float(np.clip(change_alpha_weight, 0.0, 1.0))
    if np.all(np.isnan(p_change_vec)) or change_alpha_w <= 0.0:
        change_alpha_multiplier = np.ones(len(p_chain), dtype=float)
        alpha_vec = alpha_vec_base
    else:
        change_alpha_multiplier = np.clip(
            (1.0 - change_alpha_w) + change_alpha_w * p_change_vec,
            0.0,
            1.0,
        )
        alpha_vec = np.clip(alpha_vec_base * change_alpha_multiplier, 0.0, 1.0)

    p_mix = alpha_vec[:, None] * p_chain + (1.0 - alpha_vec[:, None]) * persist_proba
    p_mix = _normalize_proba_rows(p_mix)
    pred_chain = _predict_with_class_thresholds(p_mix, class_thresholds=class_thresholds)
    conf_mix = np.max(p_mix, axis=1).astype(float)
    conf_gate_mask = conf_mix < gate_t

    change_gate_t = float(np.clip(change_gate_threshold, 0.0, 1.0))
    if np.all(np.isnan(p_change_vec)) or change_gate_t <= 0.0:
        change_gate_mask = np.zeros(len(p_chain), dtype=bool)
    else:
        change_gate_mask = p_change_vec < change_gate_t

    gate_mask = np.asarray(conf_gate_mask | change_gate_mask, dtype=bool)
    pred = np.where(gate_mask, np.asarray(y_persist, dtype=int), pred_chain).astype(int)
    return {
        "pred": pred,
        "alpha_vec": alpha_vec,
        "alpha_vec_base": alpha_vec_base,
        "change_alpha_multiplier": change_alpha_multiplier,
        "proba_mix": p_mix,
        "confidence": conf_mix,
        "p_change": p_change_vec,
        "conf_gate_mask": conf_gate_mask.astype(bool),
        "change_gate_mask": change_gate_mask.astype(bool),
        "gate_mask": gate_mask.astype(bool),
    }


def _safe_mean(x: list[float] | np.ndarray) -> float:
    """@brief Compute NaN-safe mean with empty input handling.

    @param x Numeric list or array.
    @return Mean value or NaN when empty.
    """
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return _safe_nanmean(arr)


def _aligned_split_with_persistence(
    df_all: pd.DataFrame,
    split_df: pd.DataFrame,
    horizon: int,
) -> dict[str, Any]:
    """@brief Align a split with persistence predictions.

    @param df_all Full panel with ticker/date/regime.
    @param split_df Split dataframe to align.
    @param horizon Persistence forecast horizon.
    @return Alignment payload with masks, labels, and baseline metrics.
    """
    tmp = df_all[["ticker", "date", "regime"]].copy()
    tmp["persist_pred"] = persistence_pred_from_regime(tmp, horizon=horizon, regime_col="regime")

    split_keys = split_df[["ticker", "date", "regime"]].reset_index(drop=True).copy()
    merged = split_keys.merge(
        tmp[["ticker", "date", "persist_pred"]],
        on=["ticker", "date"],
        how="left",
    )
    keep = merged["persist_pred"].notna().to_numpy()
    if int(np.sum(keep)) == 0:
        raise RuntimeError("Persistence alignment produced 0 valid rows.")

    y_true = split_keys.loc[keep, "regime"].to_numpy(dtype=int)
    y_persist = merged.loc[keep, "persist_pred"].to_numpy(dtype=int)
    m = compute_metrics(y_true, y_persist)
    return {
        "keep_mask": keep,
        "y_target_true": y_true,
        "y_target_persist": y_persist,
        "y_regime_true": y_true,
        "regime_prev": y_persist,
        "y_regime_persist": y_persist,
        "y_target_pred_persistence": y_persist,
        "y_regime_pred_persistence": y_persist,
        "target_persistence_metrics": {
            "acc": float(m.accuracy),
            "macro_f1": float(m.macro_f1),
            "weighted_f1": float(m.weighted_f1),
            "macro_recall": float(m.macro_recall),
            "high_vol_recall": _high_vol_recall(m),
        },
        "regime_persistence_metrics": {
            "acc": float(m.accuracy),
            "macro_f1": float(m.macro_f1),
            "weighted_f1": float(m.weighted_f1),
            "macro_recall": float(m.macro_recall),
            "high_vol_recall": _high_vol_recall(m),
        },
        "metrics": {
            "acc": float(m.accuracy),
            "macro_f1": float(m.macro_f1),
            "weighted_f1": float(m.weighted_f1),
            "macro_recall": float(m.macro_recall),
            "high_vol_recall": _high_vol_recall(m),
        },
    }


def _blend_predict(
    proba_chain: np.ndarray,
    y_persist: np.ndarray,
    alpha: float,
    beta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """@brief Backward-compatible blend prediction helper.

    @param proba_chain Chain model probabilities.
    @param y_persist Persistence labels.
    @param alpha Base blend alpha.
    @param beta Confidence beta.
    @return Tuple `(pred_labels, effective_alpha_per_row)`.
    """
    pred_info = _blend_strategy_predict(
        proba_chain=proba_chain,
        y_persist=y_persist,
        alpha=alpha,
        beta=beta,
        class_thresholds=None,
        gate_threshold=0.0,
    )
    return np.asarray(pred_info["pred"], dtype=int), np.asarray(pred_info["alpha_vec"], dtype=float)


def _blend_metrics(
    *,
    proba_chain: np.ndarray,
    y_true_target: np.ndarray,
    y_persist_target: np.ndarray,
    y_true_regime: np.ndarray,
    persist_target_metrics: dict[str, float],
    persist_regime_metrics: dict[str, float],
    alpha: float,
    beta: float = 0.0,
    class_thresholds: np.ndarray | tuple[float, ...] | list[float] | None = None,
    gate_threshold: float = 0.0,
    p_change: np.ndarray | None = None,
    change_gate_threshold: float = 0.0,
    change_alpha_weight: float = 0.0,
) -> dict[str, float]:
    """@brief Compute blended model metrics and deltas vs persistence.

    @param proba_chain Chain model probabilities.
    @param y_true_target Ground-truth labels in target space.
    @param y_persist_target Persistence labels in target space.
    @param y_true_regime Ground-truth labels in regime space.
    @param persist_target_metrics Persistence metrics in target space.
    @param persist_regime_metrics Persistence metrics in regime space.
    @param alpha Base blend alpha.
    @param beta Confidence beta.
    @param class_thresholds Optional class-wise thresholds.
    @param gate_threshold Confidence threshold for gating.
    @param p_change Optional sidecar p_change probabilities.
    @param change_gate_threshold Threshold below which persistence is forced by sidecar.
    @param change_alpha_weight Weight used to attenuate alpha as a function of p_change.
    @return Dictionary with metrics, deltas, and prediction artifacts.
    """
    pred_info = _blend_strategy_predict(
        proba_chain=proba_chain,
        y_persist=np.asarray(y_persist_target, dtype=int),
        alpha=alpha,
        beta=beta,
        class_thresholds=class_thresholds,
        gate_threshold=gate_threshold,
        p_change=p_change,
        change_gate_threshold=change_gate_threshold,
        change_alpha_weight=change_alpha_weight,
    )
    pred_target = np.asarray(pred_info["pred"], dtype=int)
    alpha_vec = np.asarray(pred_info["alpha_vec"], dtype=float)
    change_alpha_multiplier = np.asarray(pred_info["change_alpha_multiplier"], dtype=float)
    conf = np.asarray(pred_info["confidence"], dtype=float)
    gate_mask = np.asarray(pred_info["gate_mask"], dtype=bool)
    conf_gate_mask = np.asarray(pred_info["conf_gate_mask"], dtype=bool)
    change_gate_mask = np.asarray(pred_info["change_gate_mask"], dtype=bool)
    p_change_vec = np.asarray(pred_info["p_change"], dtype=float)

    m_target = compute_metrics(y_true_target, pred_target)
    target_high_vol_recall = _high_vol_recall(m_target)

    pred_regime = np.asarray(pred_target, dtype=int)

    m_regime = compute_metrics(y_true_regime, pred_regime)
    regime_high_vol_recall = _high_vol_recall(m_regime)
    transition_mask = np.asarray(y_true_regime, dtype=int) != np.asarray(y_persist_target, dtype=int)
    non_transition_mask = ~transition_mask
    transition_macro_f1 = _subset_macro_f1(y_true_regime, pred_regime, transition_mask)
    transition_persist_macro_f1 = _subset_macro_f1(
        y_true_regime, np.asarray(y_persist_target, dtype=int), transition_mask
    )
    non_transition_macro_f1 = _subset_macro_f1(y_true_regime, pred_regime, non_transition_mask)
    non_transition_persist_macro_f1 = _subset_macro_f1(
        y_true_regime, np.asarray(y_persist_target, dtype=int), non_transition_mask
    )

    return {
        # Legacy key names retained for backwards-compatible reports.
        "acc": float(m_regime.accuracy),
        "macro_f1": float(m_regime.macro_f1),
        "weighted_f1": float(m_regime.weighted_f1),
        "macro_recall": float(m_regime.macro_recall),
        "high_vol_recall": float(regime_high_vol_recall),
        "delta_acc_vs_persistence": float(
            m_regime.accuracy - persist_regime_metrics["acc"]
        ),
        "delta_macro_f1_vs_persistence": float(
            m_regime.macro_f1 - persist_regime_metrics["macro_f1"]
        ),
        "delta_weighted_f1_vs_persistence": float(
            m_regime.weighted_f1 - persist_regime_metrics["weighted_f1"]
        ),
        "delta_macro_recall_vs_persistence": float(
            m_regime.macro_recall - persist_regime_metrics["macro_recall"]
        ),
        "delta_high_vol_recall_vs_persistence": float(
            regime_high_vol_recall - persist_regime_metrics["high_vol_recall"]
        ),
        "target_acc": float(m_target.accuracy),
        "target_macro_f1": float(m_target.macro_f1),
        "target_weighted_f1": float(m_target.weighted_f1),
        "target_macro_recall": float(m_target.macro_recall),
        "target_high_vol_recall": float(target_high_vol_recall),
        "target_delta_acc_vs_persistence": float(
            m_target.accuracy - persist_target_metrics["acc"]
        ),
        "target_delta_macro_f1_vs_persistence": float(
            m_target.macro_f1 - persist_target_metrics["macro_f1"]
        ),
        "target_delta_weighted_f1_vs_persistence": float(
            m_target.weighted_f1 - persist_target_metrics["weighted_f1"]
        ),
        "target_delta_macro_recall_vs_persistence": float(
            m_target.macro_recall - persist_target_metrics["macro_recall"]
        ),
        "target_delta_high_vol_recall_vs_persistence": float(
            target_high_vol_recall - persist_target_metrics["high_vol_recall"]
        ),
        "mean_effective_alpha": float(np.mean(alpha_vec) if len(alpha_vec) else alpha),
        "mean_change_alpha_multiplier": float(
            np.mean(change_alpha_multiplier) if len(change_alpha_multiplier) else 1.0
        ),
        "mean_confidence": float(np.mean(conf) if len(conf) else float("nan")),
        "gating_rate": float(np.mean(gate_mask.astype(float)) if len(gate_mask) else 0.0),
        "confidence_gating_rate": float(
            np.mean(conf_gate_mask.astype(float)) if len(conf_gate_mask) else 0.0
        ),
        "change_gating_rate": float(
            np.mean(change_gate_mask.astype(float)) if len(change_gate_mask) else 0.0
        ),
        "mean_p_change": _safe_nanmean(p_change_vec),
        "selected_class_thresholds": _format_class_thresholds(
            np.ones(int(np.asarray(proba_chain).shape[1]), dtype=float)
            if class_thresholds is None
            else class_thresholds
        ),
        "selected_gate_threshold": float(np.clip(gate_threshold, 0.0, 1.0)),
        "selected_change_gate_threshold": float(np.clip(change_gate_threshold, 0.0, 1.0)),
        "selected_change_alpha_weight": float(np.clip(change_alpha_weight, 0.0, 1.0)),
        "transition_count": int(np.sum(transition_mask)),
        "transition_rate": float(np.mean(transition_mask.astype(float))),
        "transition_macro_f1": float(transition_macro_f1),
        "transition_macro_f1_persistence": float(transition_persist_macro_f1),
        "delta_transition_macro_f1_vs_persistence": float(
            transition_macro_f1 - transition_persist_macro_f1
        )
        if np.isfinite(transition_macro_f1) and np.isfinite(transition_persist_macro_f1)
        else float("nan"),
        "non_transition_count": int(np.sum(non_transition_mask)),
        "non_transition_macro_f1": float(non_transition_macro_f1),
        "non_transition_macro_f1_persistence": float(non_transition_persist_macro_f1),
        "delta_non_transition_macro_f1_vs_persistence": float(
            non_transition_macro_f1 - non_transition_persist_macro_f1
        )
        if np.isfinite(non_transition_macro_f1) and np.isfinite(non_transition_persist_macro_f1)
        else float("nan"),
        "pred_target": np.asarray(pred_target, dtype=int),
        "pred_regime": np.asarray(pred_regime, dtype=int),
    }


def _pick_best_blend_strategy(
    proba_chain: np.ndarray,
    y_true_target: np.ndarray,
    y_persist_target: np.ndarray,
    y_true_regime: np.ndarray,
    blend_alphas: tuple[float, ...],
    blend_conf_betas: tuple[float, ...],
    class_threshold_grid: tuple[tuple[float, ...], ...],
    gate_thresholds: tuple[float, ...],
    persist_target_metrics: dict[str, float],
    persist_regime_metrics: dict[str, float],
    p_change: np.ndarray | None = None,
    change_gate_thresholds: tuple[float, ...] = (0.0,),
    change_alpha_weights: tuple[float, ...] = (0.0,),
    min_non_transition_delta: float = -1.0,
) -> tuple[dict[str, Any], dict[str, float]]:
    """@brief Select best blend/gating strategy on a grid.

    @param proba_chain Chain model probabilities.
    @param y_true_target Ground-truth labels in target space.
    @param y_persist_target Persistence labels in target space.
    @param y_true_regime Ground-truth labels in regime space.
    @param blend_alphas Candidate alpha values.
    @param blend_conf_betas Candidate beta values.
    @param class_threshold_grid Candidate class-threshold vectors.
    @param gate_thresholds Candidate gating thresholds.
    @param persist_target_metrics Persistence metrics in target space.
    @param persist_regime_metrics Persistence metrics in regime space.
    @param p_change Optional sidecar p_change probabilities aligned with `proba_chain`.
    @param change_gate_thresholds Candidate p_change thresholds for persistence gating.
    @param change_alpha_weights Candidate p_change weights to attenuate alpha.
    @return Tuple `(best_config, best_metrics)`.
    """
    best_cfg = {
        "selected_alpha": float(blend_alphas[-1]),
        "selected_beta": float(blend_conf_betas[0]),
        "selected_class_thresholds": np.ones(int(np.asarray(proba_chain).shape[1]), dtype=float),
        "selected_gate_threshold": float(gate_thresholds[0]),
        "selected_change_gate_threshold": 0.0,
        "selected_change_alpha_weight": 0.0,
    }
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float, float, float] | None = None
    best_any_metrics: dict[str, float] | None = None
    best_any_score: tuple[float, float, float, float, float] | None = None
    best_any_cfg = dict(best_cfg)
    change_gate_grid = (
        tuple(float(x) for x in change_gate_thresholds)
        if p_change is not None
        else (0.0,)
    )
    change_alpha_grid = (
        tuple(float(x) for x in change_alpha_weights)
        if p_change is not None
        else (0.0,)
    )
    for alpha, beta, cls_thr, gate_thr, change_gate_thr, change_alpha_w in itertools.product(
        blend_alphas,
        blend_conf_betas,
        class_threshold_grid,
        gate_thresholds,
        change_gate_grid,
        change_alpha_grid,
    ):
        cand = _blend_metrics(
            proba_chain=proba_chain,
            y_true_target=y_true_target,
            y_persist_target=y_persist_target,
            y_true_regime=y_true_regime,
            persist_target_metrics=persist_target_metrics,
            persist_regime_metrics=persist_regime_metrics,
            alpha=float(alpha),
            beta=float(beta),
            class_thresholds=np.asarray(cls_thr, dtype=float),
            gate_threshold=float(gate_thr),
            p_change=p_change,
            change_gate_threshold=float(change_gate_thr),
            change_alpha_weight=float(change_alpha_w),
        )
        transition_delta = cand["delta_transition_macro_f1_vs_persistence"]
        if not np.isfinite(transition_delta):
            transition_delta = -1.0
        non_transition_delta = cand["delta_non_transition_macro_f1_vs_persistence"]
        if not np.isfinite(non_transition_delta):
            non_transition_delta = -1.0
        score = (
            cand["delta_macro_f1_vs_persistence"],
            float(non_transition_delta),
            float(transition_delta),
            cand["delta_high_vol_recall_vs_persistence"],
            cand["acc"],
        )
        if best_any_score is None or score > best_any_score:
            best_any_score = score
            best_any_cfg = {
                "selected_alpha": float(alpha),
                "selected_beta": float(beta),
                "selected_class_thresholds": np.asarray(cls_thr, dtype=float),
                "selected_gate_threshold": float(gate_thr),
                "selected_change_gate_threshold": float(change_gate_thr),
                "selected_change_alpha_weight": float(change_alpha_w),
            }
            best_any_metrics = cand
        if float(non_transition_delta) < float(min_non_transition_delta):
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_cfg = {
                "selected_alpha": float(alpha),
                "selected_beta": float(beta),
                "selected_class_thresholds": np.asarray(cls_thr, dtype=float),
                "selected_gate_threshold": float(gate_thr),
                "selected_change_gate_threshold": float(change_gate_thr),
                "selected_change_alpha_weight": float(change_alpha_w),
            }
            best_metrics = cand
    if best_metrics is None:
        best_cfg = best_any_cfg
        best_metrics = best_any_metrics
    assert best_metrics is not None
    return best_cfg, best_metrics


def _cache_key(
    asset: str,
    horizon: int,
    train_end: str,
    valid_end: str,
    struct_cfg: dict[str, Any],
) -> str:
    """@brief Build deterministic cache key for one fold config.

    @param asset Asset or asset-group identifier.
    @param horizon Forecast horizon.
    @param train_end Fold train end date.
    @param valid_end Fold validation end date.
    @param struct_cfg Structural chain config.
    @return Short SHA1 cache key.
    """
    payload = {
        "cache_schema": 2,
        "asset": asset,
        "horizon": int(horizon),
        "train_end": train_end,
        "valid_end": valid_end,
        **struct_cfg,
    }
    txt = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:16]


def _prepare_fold_data(
    *,
    db_path: str,
    tickers: tuple[str, ...],
    horizon: int,
    regime_bins: int,
    train_end: str,
    valid_end: str,
    struct_cfg: dict[str, Any],
    cache_dir: Path,
    use_cache: bool,
) -> dict[str, Any]:
    """@brief Build train/valid matrices and persistence alignment for one fold.

    @param db_path DuckDB path.
    @param tickers Tickers included in this run.
    @param horizon Forecast horizon.
    @param regime_bins Number of regime bins.
    @param train_end Fold train end date.
    @param valid_end Fold validation end date.
    @param struct_cfg Structural chain configuration.
    @param cache_dir Cache directory for fold artifacts.
    @param use_cache Whether to read/write fold cache.
    @return Dictionary with fold arrays, baseline labels, and metrics.
    """
    asset_name = "_".join(tickers)
    ck = _cache_key(asset_name, horizon, train_end, valid_end, struct_cfg)
    cp = cache_dir / f"fold_{ck}.npz"
    required_cache_keys = {
        "x_train",
        "y_train",
        "x_train_aligned",
        "y_train_aligned",
        "x_valid",
        "y_valid",
        "y_valid_regime",
        "regime_prev_train",
        "regime_prev_valid",
        "persist_train_pred",
        "persist_valid_pred",
        "persist_valid_regime_pred",
        "feature_cols",
        "persist_valid_acc",
        "persist_valid_macro_f1",
        "persist_valid_weighted_f1",
        "persist_valid_macro_recall",
        "persist_valid_high_vol_recall",
        "persist_valid_target_acc",
        "persist_valid_target_macro_f1",
        "persist_valid_target_weighted_f1",
        "persist_valid_target_macro_recall",
        "persist_valid_target_high_vol_recall",
        "n_features",
    }

    if use_cache and cp.exists():
        z = np.load(cp)
        if required_cache_keys.issubset(set(z.files)):
            def _scalar(name: str) -> float:
                """@brief Read a cached scalar value from an NPZ entry.

                @param name Key in NPZ file.
                @return Scalar float value.
                """
                return float(np.asarray(z[name]).reshape(-1)[0])

            return {
                "x_train": z["x_train"],
                "y_train": z["y_train"],
                "x_train_aligned": z["x_train_aligned"],
                "y_train_aligned": z["y_train_aligned"],
                "x_valid": z["x_valid"],
                "y_valid": z["y_valid"],
                "y_valid_regime": z["y_valid_regime"],
                "regime_prev_train": z["regime_prev_train"],
                "regime_prev_valid": z["regime_prev_valid"],
                "persist_train_pred": np.asarray(z["persist_train_pred"], dtype=int),
                "persist_valid_pred": np.asarray(z["persist_valid_pred"], dtype=int),
                "persist_valid_regime_pred": np.asarray(z["persist_valid_regime_pred"], dtype=int),
                "feature_cols": [str(c) for c in np.asarray(z["feature_cols"]).tolist()],
                "persist_valid_acc": _scalar("persist_valid_acc"),
                "persist_valid_macro_f1": _scalar("persist_valid_macro_f1"),
                "persist_valid_weighted_f1": _scalar("persist_valid_weighted_f1"),
                "persist_valid_macro_recall": _scalar("persist_valid_macro_recall"),
                "persist_valid_high_vol_recall": _scalar("persist_valid_high_vol_recall"),
                "persist_valid_target_acc": _scalar("persist_valid_target_acc"),
                "persist_valid_target_macro_f1": _scalar("persist_valid_target_macro_f1"),
                "persist_valid_target_weighted_f1": _scalar("persist_valid_target_weighted_f1"),
                "persist_valid_target_macro_recall": _scalar("persist_valid_target_macro_recall"),
                "persist_valid_target_high_vol_recall": _scalar(
                    "persist_valid_target_high_vol_recall"
                ),
                "n_features": int(np.asarray(z["n_features"]).reshape(-1)[0]),
            }

    dcfg = DatasetConfig(
        db_path=db_path,
        tickers=tickers,
        horizon=int(horizon),
        pooled=False,
        train_end=train_end,
        valid_end=valid_end,
        regime_bins=int(regime_bins),
        use_sarimax_garch_chain=True,
        chain_mean_model=str(struct_cfg.get("chain_mean_model", "sarimax")),
        sarimax_order=tuple(struct_cfg["sarimax_order"]),
        sarimax_chain_exog_cols=tuple(struct_cfg["sarimax_chain_exog_cols"]),
        har_target_col=str(struct_cfg.get("har_target_col", "rv_20")),
        har_lag_1=int(struct_cfg.get("har_lag_1", 1)),
        har_lag_week=int(struct_cfg.get("har_lag_week", 5)),
        har_lag_month=int(struct_cfg.get("har_lag_month", 22)),
        har_exog_cols=tuple(struct_cfg.get("har_exog_cols", ())),
        garch_p=int(struct_cfg["garch_p"]),
        garch_o=int(struct_cfg.get("garch_o", 1)),
        garch_q=int(struct_cfg["garch_q"]),
        garch_dist=str(struct_cfg["garch_dist"]),
        garch_vol=str(struct_cfg.get("garch_vol", "Garch")),
        garch_chain_agg=str(struct_cfg["garch_chain_agg"]),
        garch_scale=float(struct_cfg["garch_scale"]),
    )
    pack = make_dataset(dcfg)

    feature_cols = list(pack["feature_cols"])
    x_train_full = pack["train"][feature_cols].to_numpy(dtype=np.float32)
    y_train_full = pack["train"]["regime"].to_numpy(dtype=int)
    x_train = x_train_full
    y_train = y_train_full
    aligned_train = _aligned_split_with_persistence(
        pack["df"],
        pack["train"],
        int(horizon),
    )
    keep_train = np.asarray(aligned_train["keep_mask"], dtype=bool)
    x_train_aligned = x_train_full[keep_train]
    y_train_aligned = y_train_full[keep_train]
    regime_prev_train = np.asarray(aligned_train["regime_prev"], dtype=int)
    persist_train_pred = np.asarray(aligned_train["y_target_persist"], dtype=int)
    x_valid_full = pack["valid"][feature_cols].to_numpy(dtype=np.float32)
    y_valid_target_full = pack["valid"]["regime"].to_numpy(dtype=int)
    aligned_valid = _aligned_split_with_persistence(
        pack["df"],
        pack["valid"],
        int(horizon),
    )
    keep_valid = np.asarray(aligned_valid["keep_mask"], dtype=bool)
    x_valid = x_valid_full[keep_valid]
    y_valid = y_valid_target_full[keep_valid]
    y_valid_regime = np.asarray(aligned_valid["y_regime_true"], dtype=int)
    regime_prev_valid = np.asarray(aligned_valid["regime_prev"], dtype=int)
    persist_valid_pred = np.asarray(aligned_valid["y_target_persist"], dtype=int)
    persist_valid_regime_pred = np.asarray(aligned_valid["y_regime_persist"], dtype=int)
    p_metrics_regime = aligned_valid["regime_persistence_metrics"]
    p_metrics_target = aligned_valid["target_persistence_metrics"]

    out = {
        "x_train": x_train,
        "y_train": y_train,
        "x_train_aligned": x_train_aligned,
        "y_train_aligned": y_train_aligned,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "y_valid_regime": y_valid_regime,
        "regime_prev_train": regime_prev_train,
        "regime_prev_valid": regime_prev_valid,
        "persist_train_pred": persist_train_pred,
        "persist_valid_pred": persist_valid_pred,
        "persist_valid_regime_pred": persist_valid_regime_pred,
        "feature_cols": feature_cols,
        "persist_valid_acc": float(p_metrics_regime["acc"]),
        "persist_valid_macro_f1": float(p_metrics_regime["macro_f1"]),
        "persist_valid_weighted_f1": float(p_metrics_regime["weighted_f1"]),
        "persist_valid_macro_recall": float(p_metrics_regime["macro_recall"]),
        "persist_valid_high_vol_recall": float(p_metrics_regime["high_vol_recall"]),
        "persist_valid_target_acc": float(p_metrics_target["acc"]),
        "persist_valid_target_macro_f1": float(p_metrics_target["macro_f1"]),
        "persist_valid_target_weighted_f1": float(p_metrics_target["weighted_f1"]),
        "persist_valid_target_macro_recall": float(p_metrics_target["macro_recall"]),
        "persist_valid_target_high_vol_recall": float(p_metrics_target["high_vol_recall"]),
        "n_features": int(len(pack["feature_cols"])),
    }

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cp,
            x_train=x_train,
            y_train=y_train,
            x_train_aligned=x_train_aligned,
            y_train_aligned=y_train_aligned,
            x_valid=x_valid,
            y_valid=y_valid,
            y_valid_regime=y_valid_regime,
            regime_prev_train=regime_prev_train,
            regime_prev_valid=regime_prev_valid,
            persist_train_pred=persist_train_pred,
            persist_valid_pred=persist_valid_pred,
            persist_valid_regime_pred=persist_valid_regime_pred,
            feature_cols=np.asarray(feature_cols, dtype=str),
            persist_valid_acc=np.array([p_metrics_regime["acc"]], dtype=float),
            persist_valid_macro_f1=np.array([p_metrics_regime["macro_f1"]], dtype=float),
            persist_valid_weighted_f1=np.array([p_metrics_regime["weighted_f1"]], dtype=float),
            persist_valid_macro_recall=np.array([p_metrics_regime["macro_recall"]], dtype=float),
            persist_valid_high_vol_recall=np.array([p_metrics_regime["high_vol_recall"]], dtype=float),
            persist_valid_target_acc=np.array([p_metrics_target["acc"]], dtype=float),
            persist_valid_target_macro_f1=np.array([p_metrics_target["macro_f1"]], dtype=float),
            persist_valid_target_weighted_f1=np.array(
                [p_metrics_target["weighted_f1"]], dtype=float
            ),
            persist_valid_target_macro_recall=np.array(
                [p_metrics_target["macro_recall"]], dtype=float
            ),
            persist_valid_target_high_vol_recall=np.array(
                [p_metrics_target["high_vol_recall"]], dtype=float
            ),
            n_features=np.array([out["n_features"]], dtype=int),
        )

    return out


def _fit_eval_tabular(
    tabular_model: str,
    model_cfg: dict[str, Any],
    fold_data: dict[str, Any],
    xgb_n_jobs: int,
    use_blend: bool,
    blend_alphas: tuple[float, ...],
    blend_conf_betas: tuple[float, ...],
    calibration_method: str,
    class_threshold_grid: tuple[tuple[float, ...], ...],
    gate_thresholds: tuple[float, ...],
    min_non_transition_delta: float,
    use_gkg_change_detector: bool,
    use_gkg_change_gate: bool,
    use_gkg_change_alpha: bool,
    gkg_change_model: str,
    gkg_change_context_cols: tuple[str, ...],
    gkg_change_calibration_method: str,
    gkg_change_gate_thresholds: tuple[float, ...],
    gkg_change_alpha_weights: tuple[float, ...],
    gkg_change_blend_hook_weight: float,
) -> dict[str, Any]:
    """@brief Fit one tabular model and evaluate it on one fold.

    @param tabular_model Tabular backend name (`xgb` or `tabpfn`).
    @param model_cfg Model hyperparameters.
    @param fold_data Prepared fold payload from `_prepare_fold_data`.
    @param xgb_n_jobs Number of CPU jobs for XGB.
    @param use_blend Whether blend/gating search is enabled.
    @param blend_alphas Candidate alpha values.
    @param blend_conf_betas Candidate beta values.
    @param calibration_method Probability calibration method.
    @param class_threshold_grid Candidate class-threshold vectors.
    @param gate_thresholds Candidate gating thresholds.
    @param min_non_transition_delta Minimum accepted non-transition macro-F1 delta vs persistence.
    @param use_gkg_change_detector Whether to train/evaluate the GKG change detector sidecar.
    @param use_gkg_change_gate Whether to use sidecar `p_change` as an additional gate.
    @param use_gkg_change_alpha Whether to use sidecar `p_change` to attenuate alpha.
    @param gkg_change_model Binary model backend for change detector.
    @param gkg_change_context_cols Optional minimal context columns (e.g., regime_prev, sigma_t).
    @param gkg_change_calibration_method Calibration method for p_change.
    @param gkg_change_gate_thresholds Candidate p_change thresholds for persistence gating.
    @param gkg_change_alpha_weights Candidate p_change weights to attenuate alpha.
    @param gkg_change_blend_hook_weight Placeholder weight for future gating/reweighting.
    @return Fold-level metrics and artifacts for model selection.
    """
    model_name = str(tabular_model).lower()
    if model_name == "xgb":
        cfg = XGBConfig(
            n_estimators=int(model_cfg["n_estimators"]),
            max_depth=int(model_cfg["max_depth"]),
            learning_rate=float(model_cfg["learning_rate"]),
            subsample=float(model_cfg["subsample"]),
            colsample_bytree=float(model_cfg["colsample_bytree"]),
            min_child_weight=float(model_cfg["min_child_weight"]),
            reg_lambda=float(model_cfg["reg_lambda"]),
            n_jobs=int(xgb_n_jobs),
            random_state=42,
            seed=42,
        )
        model = make_xgb_model(cfg)
        fit_xgb(
            model,
            fold_data["x_train"],
            fold_data["y_train"],
            X_valid=fold_data["x_valid"],
            y_valid=fold_data["y_valid"],
        )
        proba_train = predict_proba_xgb(model, fold_data["x_train"])
        proba_valid = predict_proba_xgb(model, fold_data["x_valid"])
    elif model_name == "tabpfn":
        cfg = TabPFNConfig(
            n_estimators=int(model_cfg.get("n_estimators", 8)),
            softmax_temperature=float(model_cfg.get("softmax_temperature", 0.9)),
            balance_probabilities=bool(model_cfg.get("balance_probabilities", False)),
            average_before_softmax=bool(model_cfg.get("average_before_softmax", False)),
            model_path=str(model_cfg.get("model_path", "auto")),
            device=str(model_cfg.get("device", "auto")),
            ignore_pretraining_limits=bool(model_cfg.get("ignore_pretraining_limits", False)),
            inference_precision=str(model_cfg.get("inference_precision", "auto")),
            fit_mode=str(model_cfg.get("fit_mode", "fit_preprocessors")),
            memory_saving_mode=str(model_cfg.get("memory_saving_mode", "auto")),
            random_state=int(model_cfg.get("random_state", 42)),
            n_preprocessing_jobs=int(model_cfg.get("n_preprocessing_jobs", 1)),
            seed=42,
        )
        model = make_tabpfn_model(cfg)
        fit_tabpfn(
            model,
            fold_data["x_train"],
            fold_data["y_train"],
            X_valid=fold_data["x_valid"],
            y_valid=fold_data["y_valid"],
        )
        proba_train = predict_proba_tabpfn(model, fold_data["x_train"])
        proba_valid = predict_proba_tabpfn(model, fold_data["x_valid"])
    else:
        raise ValueError(f"Invalid tabular_model={tabular_model!r}")

    calibrator = _fit_probability_calibrator(
        proba=proba_train,
        y_true=np.asarray(fold_data["y_train"], dtype=int),
        method=calibration_method,
    )
    proba_valid_cal = _apply_probability_calibrator(proba_valid, calibrator)

    persist_regime_metrics = {
        "acc": fold_data["persist_valid_acc"],
        "macro_f1": fold_data["persist_valid_macro_f1"],
        "weighted_f1": fold_data["persist_valid_weighted_f1"],
        "macro_recall": fold_data["persist_valid_macro_recall"],
        "high_vol_recall": fold_data["persist_valid_high_vol_recall"],
    }
    persist_target_metrics = {
        "acc": fold_data["persist_valid_target_acc"],
        "macro_f1": fold_data["persist_valid_target_macro_f1"],
        "weighted_f1": fold_data["persist_valid_target_weighted_f1"],
        "macro_recall": fold_data["persist_valid_target_macro_recall"],
        "high_vol_recall": fold_data["persist_valid_target_high_vol_recall"],
    }

    gkg_change_eval: dict[str, Any] = {
        "enabled": False,
        "reason": "disabled",
        "gkg_change_model": str(gkg_change_model).lower(),
        "gkg_change_context_cols": ",".join(gkg_change_context_cols),
        "gkg_change_n_features": 0,
        "gkg_change_feature_names": "",
        "gkg_change_train_change_rate": float("nan"),
        "gkg_change_eval_change_rate": float("nan"),
        "gkg_change_eval_acc": float("nan"),
        "gkg_change_eval_macro_f1": float("nan"),
        "gkg_change_eval_weighted_f1": float("nan"),
        "gkg_change_eval_macro_recall": float("nan"),
        "gkg_change_eval_auc": float("nan"),
        "gkg_change_eval_brier": float("nan"),
        "gkg_change_eval_p_change_mean": float("nan"),
        "gkg_change_eval_p_change_std": float("nan"),
        "gkg_change_eval_pred_change_rate": float("nan"),
        "gkg_change_eval_p_no_change_mean": float("nan"),
        "gkg_change_calibration_method": str(gkg_change_calibration_method).lower(),
        "gkg_change_p_change_eval": np.full(len(fold_data["y_valid"]), np.nan, dtype=float),
        "gkg_change_y_eval": np.full(len(fold_data["y_valid"]), -1, dtype=int),
        "gkg_change_pred_eval": np.full(len(fold_data["y_valid"]), -1, dtype=int),
    }
    if bool(use_gkg_change_detector):
        gkg_change_eval = _fit_eval_gkg_change_detector(
            feature_cols=tuple(fold_data["feature_cols"]),
            x_train_aligned=np.asarray(fold_data["x_train_aligned"], dtype=np.float32),
            y_train_target_aligned=np.asarray(fold_data["y_train_aligned"], dtype=int),
            regime_prev_train=np.asarray(fold_data["regime_prev_train"], dtype=int),
            persist_train_pred=np.asarray(fold_data["persist_train_pred"], dtype=int),
            x_eval_aligned=np.asarray(fold_data["x_valid"], dtype=np.float32),
            y_eval_target_aligned=np.asarray(fold_data["y_valid"], dtype=int),
            regime_prev_eval=np.asarray(fold_data["regime_prev_valid"], dtype=int),
            persist_eval_pred=np.asarray(fold_data["persist_valid_pred"], dtype=int),
            model_type=str(gkg_change_model).lower(),
            calibration_method=str(gkg_change_calibration_method).lower(),
            context_cols=tuple(gkg_change_context_cols),
            xgb_n_jobs=int(xgb_n_jobs),
        )

    p_change_valid = (
        np.asarray(gkg_change_eval["gkg_change_p_change_eval"], dtype=float)
        if bool(use_gkg_change_gate or use_gkg_change_alpha)
        and bool(gkg_change_eval.get("enabled", False))
        else None
    )
    n_classes = int(np.asarray(proba_valid_cal).shape[1])
    blend_alphas_eff = blend_alphas if bool(use_blend) else (1.0,)
    blend_conf_betas_eff = blend_conf_betas if bool(use_blend) else (0.0,)
    class_threshold_grid_eff = (
        class_threshold_grid
        if bool(use_blend)
        else (tuple(np.ones(n_classes, dtype=float).tolist()),)
    )
    gate_thresholds_eff = gate_thresholds if bool(use_blend) else (0.0,)
    change_gate_thresholds_eff = (
        gkg_change_gate_thresholds if bool(use_gkg_change_gate) else (0.0,)
    )
    change_alpha_weights_eff = (
        gkg_change_alpha_weights if bool(use_gkg_change_alpha) else (0.0,)
    )
    selected_cfg, best_m = _pick_best_blend_strategy(
        proba_chain=proba_valid_cal,
        y_true_target=fold_data["y_valid"],
        y_persist_target=fold_data["persist_valid_pred"],
        y_true_regime=fold_data["y_valid_regime"],
        blend_alphas=blend_alphas_eff,
        blend_conf_betas=blend_conf_betas_eff,
        class_threshold_grid=class_threshold_grid_eff,
        gate_thresholds=gate_thresholds_eff,
        persist_target_metrics=persist_target_metrics,
        persist_regime_metrics=persist_regime_metrics,
        p_change=p_change_valid,
        change_gate_thresholds=change_gate_thresholds_eff,
        change_alpha_weights=change_alpha_weights_eff,
        min_non_transition_delta=float(min_non_transition_delta),
    )
    selected_alpha = float(selected_cfg["selected_alpha"])
    selected_beta = float(selected_cfg["selected_beta"])
    selected_class_thresholds = _parse_class_thresholds(
        selected_cfg["selected_class_thresholds"],
        n_classes=n_classes,
    )
    selected_gate_threshold = float(selected_cfg["selected_gate_threshold"])
    selected_change_gate_threshold = float(
        np.clip(selected_cfg.get("selected_change_gate_threshold", 0.0), 0.0, 1.0)
    )
    selected_change_alpha_weight = float(
        np.clip(selected_cfg.get("selected_change_alpha_weight", 0.0), 0.0, 1.0)
    )

    return {
        "selected_alpha": float(selected_alpha),
        "selected_beta": float(selected_beta),
        "selected_class_thresholds": _format_class_thresholds(selected_class_thresholds),
        "selected_gate_threshold": float(selected_gate_threshold),
        "selected_change_gate_threshold": float(selected_change_gate_threshold),
        "selected_change_alpha_weight": float(selected_change_alpha_weight),
        "calibration_method": str(calibration_method).lower(),
        "valid_acc": float(best_m["acc"]),
        "valid_macro_f1": float(best_m["macro_f1"]),
        "valid_weighted_f1": float(best_m["weighted_f1"]),
        "valid_macro_recall": float(best_m["macro_recall"]),
        "valid_high_vol_recall": float(best_m["high_vol_recall"]),
        "delta_acc_vs_persistence": float(best_m["delta_acc_vs_persistence"]),
        "delta_macro_f1_vs_persistence": float(best_m["delta_macro_f1_vs_persistence"]),
        "delta_weighted_f1_vs_persistence": float(best_m["delta_weighted_f1_vs_persistence"]),
        "delta_macro_recall_vs_persistence": float(best_m["delta_macro_recall_vs_persistence"]),
        "delta_high_vol_recall_vs_persistence": float(
            best_m["delta_high_vol_recall_vs_persistence"]
        ),
        "mean_effective_alpha": float(best_m["mean_effective_alpha"]),
        "mean_change_alpha_multiplier": float(best_m["mean_change_alpha_multiplier"]),
        "mean_confidence": float(best_m["mean_confidence"]),
        "gating_rate": float(best_m["gating_rate"]),
        "confidence_gating_rate": float(best_m["confidence_gating_rate"]),
        "change_gating_rate": float(best_m["change_gating_rate"]),
        "mean_p_change": float(best_m["mean_p_change"]),
        "transition_count": int(best_m["transition_count"]),
        "transition_rate": float(best_m["transition_rate"]),
        "transition_macro_f1": float(best_m["transition_macro_f1"]),
        "transition_macro_f1_persistence": float(best_m["transition_macro_f1_persistence"]),
        "delta_transition_macro_f1_vs_persistence": float(
            best_m["delta_transition_macro_f1_vs_persistence"]
        ),
        "non_transition_count": int(best_m["non_transition_count"]),
        "non_transition_macro_f1": float(best_m["non_transition_macro_f1"]),
        "non_transition_macro_f1_persistence": float(
            best_m["non_transition_macro_f1_persistence"]
        ),
        "delta_non_transition_macro_f1_vs_persistence": float(
            best_m["delta_non_transition_macro_f1_vs_persistence"]
        ),
        "valid_target_acc": float(best_m["target_acc"]),
        "valid_target_macro_f1": float(best_m["target_macro_f1"]),
        "valid_target_weighted_f1": float(best_m["target_weighted_f1"]),
        "valid_target_macro_recall": float(best_m["target_macro_recall"]),
        "valid_target_high_vol_recall": float(best_m["target_high_vol_recall"]),
        "target_delta_acc_vs_persistence": float(best_m["target_delta_acc_vs_persistence"]),
        "target_delta_macro_f1_vs_persistence": float(
            best_m["target_delta_macro_f1_vs_persistence"]
        ),
        "target_delta_weighted_f1_vs_persistence": float(
            best_m["target_delta_weighted_f1_vs_persistence"]
        ),
        "target_delta_macro_recall_vs_persistence": float(
            best_m["target_delta_macro_recall_vs_persistence"]
        ),
        "target_delta_high_vol_recall_vs_persistence": float(
            best_m["target_delta_high_vol_recall_vs_persistence"]
        ),
        "n_features": int(fold_data["n_features"]),
        "n_valid": int(len(fold_data["y_valid"])),
        "persist_valid_acc": float(fold_data["persist_valid_acc"]),
        "persist_valid_macro_f1": float(fold_data["persist_valid_macro_f1"]),
        "persist_valid_weighted_f1": float(fold_data["persist_valid_weighted_f1"]),
        "persist_valid_macro_recall": float(fold_data["persist_valid_macro_recall"]),
        "persist_valid_high_vol_recall": float(fold_data["persist_valid_high_vol_recall"]),
        "persist_valid_target_acc": float(fold_data["persist_valid_target_acc"]),
        "persist_valid_target_macro_f1": float(fold_data["persist_valid_target_macro_f1"]),
        "persist_valid_target_weighted_f1": float(fold_data["persist_valid_target_weighted_f1"]),
        "persist_valid_target_macro_recall": float(fold_data["persist_valid_target_macro_recall"]),
        "persist_valid_target_high_vol_recall": float(
            fold_data["persist_valid_target_high_vol_recall"]
        ),
        "gkg_change_enabled": bool(gkg_change_eval["enabled"]),
        "gkg_change_reason": str(gkg_change_eval["reason"]),
        "gkg_change_model": str(gkg_change_eval["gkg_change_model"]),
        "gkg_change_context_cols": str(gkg_change_eval["gkg_change_context_cols"]),
        "gkg_change_n_features": int(gkg_change_eval["gkg_change_n_features"]),
        "gkg_change_feature_names": str(gkg_change_eval["gkg_change_feature_names"]),
        "gkg_change_train_change_rate": float(gkg_change_eval["gkg_change_train_change_rate"]),
        "gkg_change_eval_change_rate": float(gkg_change_eval["gkg_change_eval_change_rate"]),
        "gkg_change_eval_acc": float(gkg_change_eval["gkg_change_eval_acc"]),
        "gkg_change_eval_macro_f1": float(gkg_change_eval["gkg_change_eval_macro_f1"]),
        "gkg_change_eval_weighted_f1": float(gkg_change_eval["gkg_change_eval_weighted_f1"]),
        "gkg_change_eval_macro_recall": float(gkg_change_eval["gkg_change_eval_macro_recall"]),
        "gkg_change_eval_auc": float(gkg_change_eval["gkg_change_eval_auc"]),
        "gkg_change_eval_brier": float(gkg_change_eval["gkg_change_eval_brier"]),
        "gkg_change_eval_p_change_mean": float(gkg_change_eval["gkg_change_eval_p_change_mean"]),
        "gkg_change_eval_p_change_std": float(gkg_change_eval["gkg_change_eval_p_change_std"]),
        "gkg_change_eval_pred_change_rate": float(
            gkg_change_eval["gkg_change_eval_pred_change_rate"]
        ),
        "gkg_change_eval_p_no_change_mean": float(
            gkg_change_eval["gkg_change_eval_p_no_change_mean"]
        ),
        "gkg_change_calibration_method": str(gkg_change_eval["gkg_change_calibration_method"]),
        "use_gkg_change_gate": bool(use_gkg_change_gate),
        "use_gkg_change_alpha": bool(use_gkg_change_alpha),
        "gkg_change_blend_hook_weight": float(gkg_change_blend_hook_weight),
        "proba_valid": np.asarray(proba_valid_cal, dtype=float),
        "y_valid": np.asarray(fold_data["y_valid"], dtype=int),
        "y_valid_regime": np.asarray(fold_data["y_valid_regime"], dtype=int),
        "regime_prev_valid": np.asarray(fold_data["regime_prev_valid"], dtype=int),
        "persist_valid_pred": np.asarray(fold_data["persist_valid_pred"], dtype=int),
        "persist_valid_regime_pred": np.asarray(
            fold_data["persist_valid_regime_pred"], dtype=int
        ),
        "gkg_change_p_change_eval": np.asarray(gkg_change_eval["gkg_change_p_change_eval"], dtype=float),
        "gkg_change_y_eval": np.asarray(gkg_change_eval["gkg_change_y_eval"], dtype=int),
        "gkg_change_pred_eval": np.asarray(gkg_change_eval["gkg_change_pred_eval"], dtype=int),
    }


def _official_test_compare(
    *,
    db_path: str,
    tickers: tuple[str, ...],
    horizon: int,
    regime_bins: int,
    train_end: str,
    valid_end: str,
    struct_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    tabular_model: str,
    xgb_n_jobs: int,
    use_blend: bool,
    blend_alpha: float,
    blend_beta: float,
    blend_class_thresholds: np.ndarray | tuple[float, ...] | list[float],
    blend_gate_threshold: float,
    calibration_method: str,
    use_gkg_change_detector: bool,
    use_gkg_change_gate: bool,
    use_gkg_change_alpha: bool,
    gkg_change_model: str,
    gkg_change_context_cols: tuple[str, ...],
    gkg_change_calibration_method: str,
    gkg_change_gate_threshold: float,
    gkg_change_alpha_weight: float,
    gkg_change_blend_hook_weight: float,
) -> dict[str, Any]:
    """@brief Run final official test evaluation against persistence.

    @param db_path DuckDB path.
    @param tickers Tickers included in the run.
    @param horizon Forecast horizon.
    @param regime_bins Number of regime bins.
    @param train_end Official train end date.
    @param valid_end Official validation end date.
    @param struct_cfg Selected structural chain configuration.
    @param model_cfg Selected tabular model configuration.
    @param tabular_model Tabular backend name.
    @param xgb_n_jobs Number of CPU jobs for XGB.
    @param use_blend Whether blend/gating is active.
    @param blend_alpha Selected alpha value.
    @param blend_beta Selected beta value.
    @param blend_class_thresholds Selected class thresholds.
    @param blend_gate_threshold Selected gating threshold.
    @param calibration_method Probability calibration method.
    @param use_gkg_change_detector Whether to evaluate sidecar GKG change detector.
    @param use_gkg_change_gate Whether to use the sidecar as an additional persistence gate.
    @param use_gkg_change_alpha Whether to use the sidecar to attenuate alpha.
    @param gkg_change_model Backend for change detector.
    @param gkg_change_context_cols Context columns for change detector.
    @param gkg_change_calibration_method Calibration method for p_change.
    @param gkg_change_gate_threshold Selected p_change threshold for gating.
    @param gkg_change_alpha_weight Selected p_change weight to attenuate alpha.
    @param gkg_change_blend_hook_weight Placeholder blend hook weight for future use.
    @return Final test metrics and deltas versus persistence.
    """
    dcfg = DatasetConfig(
        db_path=db_path,
        tickers=tickers,
        horizon=int(horizon),
        pooled=False,
        train_end=train_end,
        valid_end=valid_end,
        regime_bins=int(regime_bins),
        use_sarimax_garch_chain=True,
        chain_mean_model=str(struct_cfg.get("chain_mean_model", "sarimax")),
        sarimax_order=tuple(struct_cfg["sarimax_order"]),
        sarimax_chain_exog_cols=tuple(struct_cfg["sarimax_chain_exog_cols"]),
        har_target_col=str(struct_cfg.get("har_target_col", "rv_20")),
        har_lag_1=int(struct_cfg.get("har_lag_1", 1)),
        har_lag_week=int(struct_cfg.get("har_lag_week", 5)),
        har_lag_month=int(struct_cfg.get("har_lag_month", 22)),
        har_exog_cols=tuple(struct_cfg.get("har_exog_cols", ())),
        garch_p=int(struct_cfg["garch_p"]),
        garch_o=int(struct_cfg.get("garch_o", 1)),
        garch_q=int(struct_cfg["garch_q"]),
        garch_dist=str(struct_cfg["garch_dist"]),
        garch_vol=str(struct_cfg.get("garch_vol", "Garch")),
        garch_chain_agg=str(struct_cfg["garch_chain_agg"]),
        garch_scale=float(struct_cfg["garch_scale"]),
    )
    pack = make_dataset(dcfg)

    feature_cols = list(pack["feature_cols"])
    x_train = pack["train"][feature_cols].to_numpy(dtype=np.float32)
    y_train = pack["train"]["regime"].to_numpy(dtype=int)
    x_valid = pack["valid"][feature_cols].to_numpy(dtype=np.float32)
    y_valid = pack["valid"]["regime"].to_numpy(dtype=int)
    x_test = pack["test"][feature_cols].to_numpy(dtype=np.float32)
    y_test_target = pack["test"]["regime"].to_numpy(dtype=int)

    model_name = str(tabular_model).lower()
    if model_name == "xgb":
        cfg = XGBConfig(
            n_estimators=int(model_cfg["n_estimators"]),
            max_depth=int(model_cfg["max_depth"]),
            learning_rate=float(model_cfg["learning_rate"]),
            subsample=float(model_cfg["subsample"]),
            colsample_bytree=float(model_cfg["colsample_bytree"]),
            min_child_weight=float(model_cfg["min_child_weight"]),
            reg_lambda=float(model_cfg["reg_lambda"]),
            n_jobs=int(xgb_n_jobs),
            random_state=42,
            seed=42,
        )
        model = make_xgb_model(cfg)
        fit_xgb(model, x_train, y_train, X_valid=x_valid, y_valid=y_valid)
        proba_valid_for_cal = predict_proba_xgb(model, x_valid)
    elif model_name == "tabpfn":
        cfg = TabPFNConfig(
            n_estimators=int(model_cfg.get("n_estimators", 8)),
            softmax_temperature=float(model_cfg.get("softmax_temperature", 0.9)),
            balance_probabilities=bool(model_cfg.get("balance_probabilities", False)),
            average_before_softmax=bool(model_cfg.get("average_before_softmax", False)),
            model_path=str(model_cfg.get("model_path", "auto")),
            device=str(model_cfg.get("device", "auto")),
            ignore_pretraining_limits=bool(model_cfg.get("ignore_pretraining_limits", False)),
            inference_precision=str(model_cfg.get("inference_precision", "auto")),
            fit_mode=str(model_cfg.get("fit_mode", "fit_preprocessors")),
            memory_saving_mode=str(model_cfg.get("memory_saving_mode", "auto")),
            random_state=int(model_cfg.get("random_state", 42)),
            n_preprocessing_jobs=int(model_cfg.get("n_preprocessing_jobs", 1)),
            seed=42,
        )
        model = make_tabpfn_model(cfg)
        fit_tabpfn(model, x_train, y_train, X_valid=x_valid, y_valid=y_valid)
        proba_valid_for_cal = predict_proba_tabpfn(model, x_valid)
    else:
        raise ValueError(f"Invalid tabular_model={tabular_model!r}")

    aligned_test = _aligned_split_with_persistence(
        pack["df"],
        pack["test"],
        int(horizon),
    )
    aligned_train = _aligned_split_with_persistence(
        pack["df"],
        pack["train"],
        int(horizon),
    )
    keep_test = np.asarray(aligned_test["keep_mask"], dtype=bool)
    x_test_eval = x_test[keep_test]
    y_test_target_eval = y_test_target[keep_test]
    y_test_regime = np.asarray(aligned_test["y_regime_true"], dtype=int)
    regime_prev_test = np.asarray(aligned_test["regime_prev"], dtype=int)
    y_persist_test_target = np.asarray(aligned_test["y_target_persist"], dtype=int)
    keep_train_change = np.asarray(aligned_train["keep_mask"], dtype=bool)
    x_train_aligned_change = x_train[keep_train_change]
    y_train_aligned_change = y_train[keep_train_change]
    regime_prev_train_change = np.asarray(aligned_train["regime_prev"], dtype=int)
    y_persist_train_change = np.asarray(aligned_train["y_target_persist"], dtype=int)
    p_metrics_regime = aligned_test["regime_persistence_metrics"]
    p_metrics_target = aligned_test["target_persistence_metrics"]

    if model_name == "xgb":
        proba_test = predict_proba_xgb(model, x_test_eval)
    else:
        proba_test = predict_proba_tabpfn(model, x_test_eval)
    calibrator = _fit_probability_calibrator(
        proba=proba_valid_for_cal,
        y_true=y_valid,
        method=calibration_method,
    )
    proba_test_cal = _apply_probability_calibrator(proba_test, calibrator)

    class_thresholds = _parse_class_thresholds(
        blend_class_thresholds, n_classes=int(np.asarray(proba_test_cal).shape[1])
    )
    if bool(use_blend):
        alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        beta = float(np.clip(blend_beta, 0.0, 1.0))
    else:
        alpha = 1.0
        beta = 0.0
    gkg_change_test_eval: dict[str, Any] = {
        "enabled": False,
        "reason": "disabled",
        "gkg_change_model": str(gkg_change_model).lower(),
        "gkg_change_context_cols": ",".join(gkg_change_context_cols),
        "gkg_change_n_features": 0,
        "gkg_change_feature_names": "",
        "gkg_change_eval_change_rate": float("nan"),
        "gkg_change_eval_acc": float("nan"),
        "gkg_change_eval_macro_f1": float("nan"),
        "gkg_change_eval_weighted_f1": float("nan"),
        "gkg_change_eval_macro_recall": float("nan"),
        "gkg_change_eval_auc": float("nan"),
        "gkg_change_eval_brier": float("nan"),
        "gkg_change_eval_p_change_mean": float("nan"),
        "gkg_change_eval_p_change_std": float("nan"),
        "gkg_change_eval_pred_change_rate": float("nan"),
        "gkg_change_eval_p_no_change_mean": float("nan"),
        "gkg_change_calibration_method": str(gkg_change_calibration_method).lower(),
    }
    if bool(use_gkg_change_detector):
        gkg_change_test_eval = _fit_eval_gkg_change_detector(
            feature_cols=tuple(feature_cols),
            x_train_aligned=np.asarray(x_train_aligned_change, dtype=np.float32),
            y_train_target_aligned=np.asarray(y_train_aligned_change, dtype=int),
            regime_prev_train=np.asarray(regime_prev_train_change, dtype=int),
            persist_train_pred=np.asarray(y_persist_train_change, dtype=int),
            x_eval_aligned=np.asarray(x_test_eval, dtype=np.float32),
            y_eval_target_aligned=np.asarray(y_test_target_eval, dtype=int),
            regime_prev_eval=np.asarray(regime_prev_test, dtype=int),
            persist_eval_pred=np.asarray(y_persist_test_target, dtype=int),
            model_type=str(gkg_change_model).lower(),
            calibration_method=str(gkg_change_calibration_method).lower(),
            context_cols=tuple(gkg_change_context_cols),
            xgb_n_jobs=int(xgb_n_jobs),
        )

    p_change_test = (
        np.asarray(gkg_change_test_eval.get("gkg_change_p_change_eval"), dtype=float)
        if bool(use_gkg_change_gate or use_gkg_change_alpha)
        and bool(gkg_change_test_eval.get("enabled", False))
        else None
    )
    pred_info = _blend_strategy_predict(
        proba_chain=proba_test_cal,
        y_persist=y_persist_test_target,
        alpha=alpha,
        beta=beta,
        class_thresholds=class_thresholds,
        gate_threshold=float(blend_gate_threshold),
        p_change=p_change_test,
        change_gate_threshold=float(gkg_change_gate_threshold),
        change_alpha_weight=float(gkg_change_alpha_weight),
    )
    pred_test = np.asarray(pred_info["pred"], dtype=int)
    alpha_vec = np.asarray(pred_info["alpha_vec"], dtype=float)
    change_alpha_multiplier = np.asarray(pred_info["change_alpha_multiplier"], dtype=float)
    conf_test = np.asarray(pred_info["confidence"], dtype=float)
    gate_mask = np.asarray(pred_info["gate_mask"], dtype=bool)
    conf_gate_mask = np.asarray(pred_info["conf_gate_mask"], dtype=bool)
    change_gate_mask = np.asarray(pred_info["change_gate_mask"], dtype=bool)
    p_change_vec = np.asarray(pred_info["p_change"], dtype=float)

    m_chain_target = compute_metrics(y_test_target_eval, pred_test)
    chain_target_high_vol_recall = _high_vol_recall(m_chain_target)
    pred_test_regime = np.asarray(pred_test, dtype=int)
    m_chain_regime = compute_metrics(y_test_regime, pred_test_regime)
    chain_regime_high_vol_recall = _high_vol_recall(m_chain_regime)
    transition_mask = (
        np.asarray(y_test_regime, dtype=int) != np.asarray(y_persist_test_target, dtype=int)
    )
    non_transition_mask = ~transition_mask
    transition_macro_f1 = _subset_macro_f1(y_test_regime, pred_test_regime, transition_mask)
    transition_persist_macro_f1 = _subset_macro_f1(
        y_test_regime, np.asarray(y_persist_test_target, dtype=int), transition_mask
    )
    non_transition_macro_f1 = _subset_macro_f1(
        y_test_regime, pred_test_regime, non_transition_mask
    )
    non_transition_persist_macro_f1 = _subset_macro_f1(
        y_test_regime, np.asarray(y_persist_test_target, dtype=int), non_transition_mask
    )

    return {
        "selected_blend_alpha": alpha,
        "selected_blend_beta": beta,
        "selected_class_thresholds": _format_class_thresholds(class_thresholds),
        "selected_gate_threshold": float(np.clip(blend_gate_threshold, 0.0, 1.0)),
        "selected_change_gate_threshold": float(np.clip(gkg_change_gate_threshold, 0.0, 1.0)),
        "selected_change_alpha_weight": float(np.clip(gkg_change_alpha_weight, 0.0, 1.0)),
        "calibration_method": str(calibration_method).lower(),
        "mean_effective_alpha_test": float(np.mean(alpha_vec) if len(alpha_vec) else alpha),
        "mean_change_alpha_multiplier_test": float(
            np.mean(change_alpha_multiplier) if len(change_alpha_multiplier) else 1.0
        ),
        "mean_confidence_test": float(np.mean(conf_test) if len(conf_test) else float("nan")),
        "gating_rate_test": float(np.mean(gate_mask.astype(float)) if len(gate_mask) else 0.0),
        "confidence_gating_rate_test": float(
            np.mean(conf_gate_mask.astype(float)) if len(conf_gate_mask) else 0.0
        ),
        "change_gating_rate_test": float(
            np.mean(change_gate_mask.astype(float)) if len(change_gate_mask) else 0.0
        ),
        "mean_p_change_test": _safe_nanmean(p_change_vec),
        "chain_test_acc": float(m_chain_regime.accuracy),
        "chain_test_macro_f1": float(m_chain_regime.macro_f1),
        "chain_test_weighted_f1": float(m_chain_regime.weighted_f1),
        "chain_test_macro_recall": float(m_chain_regime.macro_recall),
        "chain_test_high_vol_recall": float(chain_regime_high_vol_recall),
        "persistence_test_acc": float(p_metrics_regime["acc"]),
        "persistence_test_macro_f1": float(p_metrics_regime["macro_f1"]),
        "persistence_test_weighted_f1": float(p_metrics_regime["weighted_f1"]),
        "persistence_test_macro_recall": float(p_metrics_regime["macro_recall"]),
        "persistence_test_high_vol_recall": float(p_metrics_regime["high_vol_recall"]),
        "delta_test_acc_vs_persistence": float(
            m_chain_regime.accuracy - p_metrics_regime["acc"]
        ),
        "delta_test_macro_f1_vs_persistence": float(
            m_chain_regime.macro_f1 - p_metrics_regime["macro_f1"]
        ),
        "delta_test_weighted_f1_vs_persistence": float(
            m_chain_regime.weighted_f1 - p_metrics_regime["weighted_f1"]
        ),
        "delta_test_macro_recall_vs_persistence": float(
            m_chain_regime.macro_recall - p_metrics_regime["macro_recall"]
        ),
        "delta_test_high_vol_recall_vs_persistence": float(
            chain_regime_high_vol_recall - p_metrics_regime["high_vol_recall"]
        ),
        "chain_test_target_acc": float(m_chain_target.accuracy),
        "chain_test_target_macro_f1": float(m_chain_target.macro_f1),
        "chain_test_target_weighted_f1": float(m_chain_target.weighted_f1),
        "chain_test_target_macro_recall": float(m_chain_target.macro_recall),
        "chain_test_target_high_vol_recall": float(chain_target_high_vol_recall),
        "persistence_test_target_acc": float(p_metrics_target["acc"]),
        "persistence_test_target_macro_f1": float(p_metrics_target["macro_f1"]),
        "persistence_test_target_weighted_f1": float(p_metrics_target["weighted_f1"]),
        "persistence_test_target_macro_recall": float(p_metrics_target["macro_recall"]),
        "persistence_test_target_high_vol_recall": float(p_metrics_target["high_vol_recall"]),
        "target_delta_test_acc_vs_persistence": float(
            m_chain_target.accuracy - p_metrics_target["acc"]
        ),
        "target_delta_test_macro_f1_vs_persistence": float(
            m_chain_target.macro_f1 - p_metrics_target["macro_f1"]
        ),
        "target_delta_test_weighted_f1_vs_persistence": float(
            m_chain_target.weighted_f1 - p_metrics_target["weighted_f1"]
        ),
        "target_delta_test_macro_recall_vs_persistence": float(
            m_chain_target.macro_recall - p_metrics_target["macro_recall"]
        ),
        "target_delta_test_high_vol_recall_vs_persistence": float(
            chain_target_high_vol_recall - p_metrics_target["high_vol_recall"]
        ),
        "transition_count_test": int(np.sum(transition_mask)),
        "transition_rate_test": float(np.mean(transition_mask.astype(float))),
        "transition_macro_f1_test": float(transition_macro_f1),
        "transition_macro_f1_persistence_test": float(transition_persist_macro_f1),
        "delta_transition_macro_f1_vs_persistence_test": float(
            transition_macro_f1 - transition_persist_macro_f1
        )
        if np.isfinite(transition_macro_f1) and np.isfinite(transition_persist_macro_f1)
        else float("nan"),
        "non_transition_count_test": int(np.sum(non_transition_mask)),
        "non_transition_macro_f1_test": float(non_transition_macro_f1),
        "non_transition_macro_f1_persistence_test": float(non_transition_persist_macro_f1),
        "delta_non_transition_macro_f1_vs_persistence_test": float(
            non_transition_macro_f1 - non_transition_persist_macro_f1
        )
        if np.isfinite(non_transition_macro_f1) and np.isfinite(non_transition_persist_macro_f1)
        else float("nan"),
        "gkg_change_enabled_test": bool(gkg_change_test_eval.get("enabled", False)),
        "gkg_change_reason_test": str(gkg_change_test_eval.get("reason", "")),
        "gkg_change_model_test": str(gkg_change_test_eval.get("gkg_change_model", "")),
        "gkg_change_context_cols_test": str(
            gkg_change_test_eval.get("gkg_change_context_cols", "")
        ),
        "gkg_change_n_features_test": int(gkg_change_test_eval.get("gkg_change_n_features", 0)),
        "gkg_change_macro_f1_test": float(
            gkg_change_test_eval.get("gkg_change_eval_macro_f1", np.nan)
        ),
        "gkg_change_auc_test": float(gkg_change_test_eval.get("gkg_change_eval_auc", np.nan)),
        "gkg_change_brier_test": float(
            gkg_change_test_eval.get("gkg_change_eval_brier", np.nan)
        ),
        "gkg_change_p_change_mean_test": float(
            gkg_change_test_eval.get("gkg_change_eval_p_change_mean", np.nan)
        ),
        "gkg_change_pred_change_rate_test": float(
            gkg_change_test_eval.get("gkg_change_eval_pred_change_rate", np.nan)
        ),
        "gkg_change_calibration_method_test": str(
            gkg_change_test_eval.get("gkg_change_calibration_method", "")
        ),
        "use_gkg_change_alpha": bool(use_gkg_change_alpha),
        "gkg_change_blend_hook_weight": float(gkg_change_blend_hook_weight),
        "n_test_eval": int(len(y_test_regime)),
        "n_features": int(len(feature_cols)),
    }


def main() -> int:
    """@brief CLI entrypoint for walk-forward selection and final evaluation.

    @return Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Walk-forward model selection for chain SARIMAX->GARCH->tabular."
    )
    parser.add_argument("--config_features", default="config/features.yaml")
    parser.add_argument("--config_sources", default="config/datasources.yaml")
    parser.add_argument(
        "--chain_variants_config",
        default=DEFAULT_CHAIN_VARIANTS_CONFIG,
        help="YAML with chain (SARIMAX+GARCH) structural variants.",
    )
    parser.add_argument(
        "--tabpfn_variants_config",
        default=DEFAULT_TABPFN_VARIANTS_CONFIG,
        help="YAML with TabPFN model variants.",
    )
    parser.add_argument("--horizon", type=int, default=20, choices=[5, 20])
    parser.add_argument(
        "--tabular_model",
        default="xgb",
        choices=["xgb", "tabpfn"],
        help="Tabular classifier used after SARIMAX->GARCH chain.",
    )
    parser.add_argument("--tickers", nargs="+", default=["^GSPC", "BTC-USD", "TLT"])
    parser.add_argument("--asset", default=None, help="Run only for a single asset/ticker.")
    parser.add_argument("--min_train_end", default="2018-12-31")
    parser.add_argument("--max_valid_end", default="2023-12-31")
    parser.add_argument("--valid_months", type=int, default=12)
    parser.add_argument("--step_months", type=int, default=6)
    parser.add_argument("--outdir", default="runs/walk_forward_chain_xgb")
    parser.add_argument("--max_experiments", type=int, default=0)
    parser.add_argument("--max_struct_configs", type=int, default=0)
    parser.add_argument("--max_xgb_configs", type=int, default=0)
    parser.add_argument(
        "--max_model_configs",
        type=int,
        default=0,
        help="Generic cap for tabular configurations (overrides --max_xgb_configs if >0).",
    )
    parser.add_argument("--min_folds_ok", type=int, default=2)
    parser.add_argument("--min_positive_rate", type=float, default=0.5)
    parser.add_argument(
        "--min_non_transition_delta",
        type=float,
        default=-0.02,
        help="Minimum mean delta of non-transition macro F1 (chain - persistence).",
    )
    parser.add_argument(
        "--non_transition_penalty_lambda",
        type=float,
        default=1.0,
        help="Penalty weight when mean non-transition delta falls below --min_non_transition_delta.",
    )
    parser.add_argument(
        "--min_high_vol_recall_delta",
        type=float,
        default=0.0,
        help="Minimum mean delta of high-vol recall (chain - persistence) required in selection.",
    )
    parser.add_argument("--stability_lambda", type=float, default=1.0)
    parser.add_argument("--positive_rate_lambda", type=float, default=0.5)
    parser.add_argument("--prune_after_folds", type=int, default=2)
    parser.add_argument("--prune_delta_threshold", type=float, default=-0.03)
    parser.add_argument("--xgb_n_jobs", type=int, default=1)
    parser.add_argument("--tabpfn_device", default="auto")
    parser.add_argument("--tabpfn_n_preprocessing_jobs", type=int, default=1)
    parser.add_argument(
        "--tabpfn_model_version",
        default="v2",
        choices=["v2", "v2.5"],
        help="TabPFN checkpoint family to use.",
    )
    parser.add_argument(
        "--use_gkg_change_detector",
        action="store_true",
        help="Enable a sidecar binary detector (change/no-change) using only GKG-derived features.",
    )
    parser.add_argument(
        "--gkg_change_model",
        default="logit",
        choices=["logit", "xgb", "tabpfn"],
        help="Backend for the GKG change detector (tabpfn reserved for future extension).",
    )
    parser.add_argument(
        "--gkg_change_context_cols",
        default="regime_prev",
        help="Comma-separated optional context columns for change detector: regime_prev,sigma_t",
    )
    parser.add_argument(
        "--gkg_change_calibration_method",
        default="none",
        choices=["none", "platt", "isotonic"],
        help="Calibration method for p_change probabilities.",
    )
    parser.add_argument(
        "--gkg_change_blend_hook_weight",
        type=float,
        default=0.0,
        help="Reserved weight for future blend/gating integration using p_change.",
    )
    parser.add_argument(
        "--use_gkg_change_gate",
        action="store_true",
        help="Use sidecar p_change as an additional gate: low p_change routes to persistence.",
    )
    parser.add_argument(
        "--use_gkg_change_alpha",
        action="store_true",
        help="Use sidecar p_change to attenuate alpha directly inside the chain/persistence blend.",
    )
    parser.add_argument(
        "--gkg_change_gate_thresholds",
        default="0.0,0.40,0.50,0.60,0.70",
        help="Comma-separated p_change thresholds for persistence gating.",
    )
    parser.add_argument(
        "--gkg_change_alpha_weights",
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated weights used to attenuate alpha as alpha*((1-w)+w*p_change).",
    )
    parser.add_argument(
        "--gkg_change_alpha_weights_by_asset",
        default="",
        help=(
            "Optional semicolon-separated asset overrides, e.g. "
            "'BTC-USD=0.0,0.1,0.2;TLT=0.0,0.05,0.1'. "
            "When present, each asset uses its own alpha-weight grid."
        ),
    )
    parser.add_argument("--use_blend", action="store_true")
    parser.add_argument(
        "--blend_alphas",
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated blend weights alpha for chain probs in alpha*chain + (1-alpha)*persistence.",
    )
    parser.add_argument(
        "--blend_conf_betas",
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated confidence blend betas. beta=0 means constant alpha; beta>0 adapts alpha by chain confidence.",
    )
    parser.add_argument(
        "--calibration_method",
        default="none",
        choices=["none", "platt", "isotonic"],
        help="Probability calibration method applied before blend/gating.",
    )
    parser.add_argument(
        "--gate_thresholds",
        default="0.0,0.55,0.60,0.65,0.70,0.75",
        help="Comma-separated confidence thresholds for persistence-aware gating.",
    )
    parser.add_argument(
        "--class_threshold_grid",
        default="1,1,1;0.95,1,1.05;1,0.95,1.05;1.05,0.95,1",
        help="Semicolon-separated class-threshold vectors, each with 3 comma-separated positive values.",
    )
    parser.add_argument(
        "--transition_bonus_lambda",
        type=float,
        default=0.0,
        help="Bonus weight for mean delta transition macro F1 in robust score.",
    )
    parser.add_argument("--disable_cache", action="store_true")
    parser.add_argument("--per_ticker", action="store_true")
    parser.add_argument(
        "--grid_profile",
        default="promising",
        choices=["promising", "full"],
        help="Hyperparameter grid profile: compact promising set or broader full set.",
    )
    # --- MLflow tracking ---
    parser.add_argument(
        "--use_mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking.",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        default="file:./mlruns",
        help="MLflow tracking store URI.",
    )
    parser.add_argument(
        "--mlflow_experiment",
        default="walk_forward_chain_tab",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow_parent_run_name",
        default="",
        help="Override parent run name (auto-generated timestamp if empty).",
    )
    parser.add_argument(
        "--mlflow_strict",
        action="store_true",
        help="Abort the run if any MLflow tracking call fails.",
    )
    parser.add_argument(
        "--mlflow_tags_json",
        default="",
        help="Extra MLflow tags as a JSON string, e.g. '{\"env\": \"dev\"}'.",
    )
    args = parser.parse_args()
    # --- Build MLflow config ---
    _mlflow_extra_tags: dict[str, str] = {}
    if args.mlflow_tags_json:
        try:
            _mlflow_extra_tags = json.loads(args.mlflow_tags_json)
        except json.JSONDecodeError as _e:
            raise SystemExit(f"Invalid --mlflow_tags_json: {_e}") from _e
    mlflow_cfg = MlflowConfig(
        enabled=bool(args.use_mlflow),
        tracking_uri=str(args.mlflow_tracking_uri),
        experiment_name=str(args.mlflow_experiment),
        parent_run_name=str(args.mlflow_parent_run_name),
        strict=bool(args.mlflow_strict),
        extra_tags=_mlflow_extra_tags,
    )
    blend_alphas = _parse_blend_alphas(args.blend_alphas)
    blend_conf_betas = _parse_blend_conf_betas(args.blend_conf_betas)
    gate_thresholds = _parse_gate_thresholds(args.gate_thresholds)
    gkg_change_gate_thresholds = _parse_gkg_change_gate_thresholds(
        args.gkg_change_gate_thresholds
    )
    gkg_change_alpha_weights = _parse_gkg_change_alpha_weights(
        args.gkg_change_alpha_weights
    )
    gkg_change_alpha_weights_by_asset = _parse_gkg_change_alpha_weights_by_asset(
        args.gkg_change_alpha_weights_by_asset
    )
    class_threshold_grid = _parse_class_threshold_grid(args.class_threshold_grid, n_classes=3)
    gkg_change_context_cols = _parse_csv_list(args.gkg_change_context_cols)
    valid_change_context = {"regime_prev", "sigma_t"}
    invalid_change_context = [c for c in gkg_change_context_cols if c not in valid_change_context]
    if invalid_change_context:
        raise SystemExit(
            "Invalid --gkg_change_context_cols entries: "
            f"{invalid_change_context}. Allowed: {sorted(valid_change_context)}"
        )
    if bool(args.use_gkg_change_detector) and str(args.gkg_change_model).lower() == "tabpfn":
        raise SystemExit(
            "--gkg_change_model=tabpfn is reserved for future extension and is not implemented yet. "
            "Use logit or xgb."
        )
    if bool(args.use_gkg_change_gate) and not bool(args.use_gkg_change_detector):
        raise SystemExit(
            "--use_gkg_change_gate requires --use_gkg_change_detector because it gates on p_change."
        )
    if bool(args.use_gkg_change_alpha) and not bool(args.use_gkg_change_detector):
        raise SystemExit(
            "--use_gkg_change_alpha requires --use_gkg_change_detector because it reweights alpha with p_change."
        )

    if str(args.tabular_model) == "tabpfn" and not os.environ.get("TABPFN_MODEL_CACHE_DIR"):
        tabpfn_cache_dir = Path(args.outdir) / ".tabpfn_cache"
        tabpfn_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TABPFN_MODEL_CACHE_DIR"] = str(tabpfn_cache_dir)
    if str(args.tabular_model) == "tabpfn":
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

    features_cfg = load_yaml(args.config_features)
    sources_cfg = load_yaml(args.config_sources)
    db_path = sources_cfg["db"]["path"]
    regime_bins = int(features_cfg["targets"]["regime_bins"])
    tickers = tuple(args.tickers)
    if args.asset:
        tickers = (str(args.asset),)

    structural_cfgs = make_structural_configs(
        args.grid_profile, variants_path=str(args.chain_variants_config)
    )
    if str(args.tabular_model) == "xgb":
        xgb_cfgs = make_xgb_configs(args.grid_profile)
    elif str(args.tabular_model) == "tabpfn":
        xgb_cfgs = make_tabpfn_configs(
            args.grid_profile,
            device=str(args.tabpfn_device),
            n_preprocessing_jobs=int(args.tabpfn_n_preprocessing_jobs),
            model_version=str(args.tabpfn_model_version),
            variants_path=str(args.tabpfn_variants_config),
        )
    else:
        raise SystemExit(f"Invalid --tabular_model={args.tabular_model!r}")

    if int(args.max_struct_configs) > 0:
        structural_cfgs = structural_cfgs[: int(args.max_struct_configs)]
    cfg_cap = int(args.max_model_configs) if int(args.max_model_configs) > 0 else int(args.max_xgb_configs)
    if cfg_cap > 0:
        xgb_cfgs = xgb_cfgs[:cfg_cap]
    if int(args.max_experiments) > 0:
        # Backward compatibility: cap structural configs first.
        structural_cfgs = structural_cfgs[: int(args.max_experiments)]

    if args.asset:
        groups = [(str(args.asset), tickers)]
    else:
        groups = (
            [(ticker, (ticker,)) for ticker in tickers]
            if bool(args.per_ticker)
            else [("all", tickers)]
        )

    global_best_rows: list[dict[str, Any]] = []
    final_compare_rows: list[dict[str, Any]] = []

    for asset_name, group_tickers in groups:
        asset_gkg_change_alpha_weights = _resolve_asset_specific_grid(
            default_grid=gkg_change_alpha_weights,
            asset_overrides=gkg_change_alpha_weights_by_asset,
            asset_name=str(asset_name),
            tickers=tuple(group_tickers),
        )
        folds = build_folds(
            min_train_end=args.min_train_end,
            max_valid_end=args.max_valid_end,
            valid_months=int(args.valid_months),
            step_months=int(args.step_months),
        )
        if not folds:
            raise SystemExit(f"No folds generated for asset={asset_name}. Check date range arguments.")

        outdir = Path(args.outdir) / f"h{args.horizon}" / f"asset_{asset_name}"
        cache_dir = outdir / "cache"
        outdir.mkdir(parents=True, exist_ok=True)

        fold_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        for s_idx, s_cfg in enumerate(structural_cfgs):
            combo_metrics: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(xgb_cfgs))}
            active_xgb = set(range(len(xgb_cfgs)))
            pruned_at: dict[int, int | None] = {i: None for i in range(len(xgb_cfgs))}

            for fold_idx, (train_end, valid_end) in enumerate(folds):
                fold_data = _prepare_fold_data(
                    db_path=db_path,
                    tickers=group_tickers,
                    horizon=int(args.horizon),
                    regime_bins=regime_bins,
                    train_end=train_end,
                    valid_end=valid_end,
                    struct_cfg=s_cfg,
                    cache_dir=cache_dir,
                    use_cache=not bool(args.disable_cache),
                )

                for x_idx in sorted(active_xgb):
                    try:
                        res = _fit_eval_tabular(
                            tabular_model=str(args.tabular_model),
                            model_cfg=xgb_cfgs[x_idx],
                            fold_data=fold_data,
                            xgb_n_jobs=int(args.xgb_n_jobs),
                            use_blend=bool(args.use_blend),
                            blend_alphas=blend_alphas,
                            blend_conf_betas=blend_conf_betas,
                            calibration_method=str(args.calibration_method),
                            class_threshold_grid=class_threshold_grid,
                            gate_thresholds=gate_thresholds,
                            min_non_transition_delta=float(args.min_non_transition_delta),
                            use_gkg_change_detector=bool(args.use_gkg_change_detector),
                            use_gkg_change_gate=bool(args.use_gkg_change_gate),
                            use_gkg_change_alpha=bool(args.use_gkg_change_alpha),
                            gkg_change_model=str(args.gkg_change_model),
                            gkg_change_context_cols=gkg_change_context_cols,
                            gkg_change_calibration_method=str(
                                args.gkg_change_calibration_method
                            ),
                            gkg_change_gate_thresholds=gkg_change_gate_thresholds,
                            gkg_change_alpha_weights=asset_gkg_change_alpha_weights,
                            gkg_change_blend_hook_weight=float(
                                args.gkg_change_blend_hook_weight
                            ),
                        )
                        combo_metrics[x_idx].append(res)
                        res_row = {
                            k: v
                            for k, v in res.items()
                            if k
                            not in {
                                "proba_valid",
                                "y_valid",
                                "y_valid_regime",
                                "regime_prev_valid",
                                "persist_valid_pred",
                                "persist_valid_regime_pred",
                                "gkg_change_p_change_eval",
                                "gkg_change_y_eval",
                                "gkg_change_pred_eval",
                            }
                        }
                        fold_rows.append(
                            {
                                "asset": asset_name,
                                "tickers": "|".join(group_tickers),
                                "tabular_model": str(args.tabular_model),
                                "struct_id": s_idx,
                                "model_id": x_idx,
                                "xgb_id": x_idx,
                                "fold_id": fold_idx,
                                "train_end": train_end,
                                "valid_end": valid_end,
                                **s_cfg,
                                **xgb_cfgs[x_idx],
                                **res_row,
                            }
                        )
                    except Exception as e:
                        fold_rows.append(
                            {
                                "asset": asset_name,
                                "tickers": "|".join(group_tickers),
                                "tabular_model": str(args.tabular_model),
                                "struct_id": s_idx,
                                "model_id": x_idx,
                                "xgb_id": x_idx,
                                "fold_id": fold_idx,
                                "train_end": train_end,
                                "valid_end": valid_end,
                                **s_cfg,
                                **xgb_cfgs[x_idx],
                                "error": repr(e),
                            }
                        )

                if int(args.prune_after_folds) > 0:
                    for x_idx in list(active_xgb):
                        rows = combo_metrics[x_idx]
                        if len(rows) < int(args.prune_after_folds):
                            continue
                        mean_delta = float(np.mean([r["delta_macro_f1_vs_persistence"] for r in rows]))
                        mean_non_transition_delta = float(
                            np.mean(
                                [
                                    r["delta_non_transition_macro_f1_vs_persistence"]
                                    for r in rows
                                ]
                            )
                        )
                        mean_high_vol_recall_delta = float(
                            np.mean([r["delta_high_vol_recall_vs_persistence"] for r in rows])
                        )
                        if (
                            mean_delta < float(args.prune_delta_threshold)
                            or mean_non_transition_delta < float(args.min_non_transition_delta)
                            or mean_high_vol_recall_delta < float(args.min_high_vol_recall_delta)
                        ):
                            active_xgb.remove(x_idx)
                            pruned_at[x_idx] = fold_idx

                if not active_xgb:
                    break

            for x_idx, rows in combo_metrics.items():
                if not rows:
                    continue
                alpha_series_fold = np.array([r["selected_alpha"] for r in rows], dtype=float)
                beta_series_fold = np.array([r["selected_beta"] for r in rows], dtype=float)
                eff_alpha_series_fold = np.array([r["mean_effective_alpha"] for r in rows], dtype=float)

                oof_proba = np.vstack([np.asarray(r["proba_valid"], dtype=float) for r in rows])
                oof_y_target = np.concatenate(
                    [np.asarray(r["y_valid"], dtype=int) for r in rows]
                ).astype(int)
                oof_y_regime = np.concatenate(
                    [np.asarray(r["y_valid_regime"], dtype=int) for r in rows]
                ).astype(int)
                oof_persist_target = np.concatenate(
                    [np.asarray(r["persist_valid_pred"], dtype=int) for r in rows]
                ).astype(int)
                oof_persist_regime = np.concatenate(
                    [np.asarray(r["persist_valid_regime_pred"], dtype=int) for r in rows]
                ).astype(int)
                oof_p_change = (
                    np.concatenate(
                        [np.asarray(r["gkg_change_p_change_eval"], dtype=float) for r in rows]
                    ).astype(float)
                    if bool(args.use_gkg_change_gate or args.use_gkg_change_alpha)
                    else None
                )

                m_oof_persist_target = compute_metrics(oof_y_target, oof_persist_target)
                oof_persist_target_metrics = {
                    "acc": float(m_oof_persist_target.accuracy),
                    "macro_f1": float(m_oof_persist_target.macro_f1),
                    "weighted_f1": float(m_oof_persist_target.weighted_f1),
                    "macro_recall": float(m_oof_persist_target.macro_recall),
                    "high_vol_recall": _high_vol_recall(m_oof_persist_target),
                }
                m_oof_persist_regime = compute_metrics(oof_y_regime, oof_persist_regime)
                oof_persist_regime_metrics = {
                    "acc": float(m_oof_persist_regime.accuracy),
                    "macro_f1": float(m_oof_persist_regime.macro_f1),
                    "weighted_f1": float(m_oof_persist_regime.weighted_f1),
                    "macro_recall": float(m_oof_persist_regime.macro_recall),
                    "high_vol_recall": _high_vol_recall(m_oof_persist_regime),
                }

                oof_n_classes = int(np.asarray(oof_proba).shape[1])
                oof_selected_cfg, oof_best_m = _pick_best_blend_strategy(
                    proba_chain=oof_proba,
                    y_true_target=oof_y_target,
                    y_persist_target=oof_persist_target,
                    y_true_regime=oof_y_regime,
                    blend_alphas=blend_alphas if bool(args.use_blend) else (1.0,),
                    blend_conf_betas=blend_conf_betas if bool(args.use_blend) else (0.0,),
                    class_threshold_grid=(
                        class_threshold_grid
                        if bool(args.use_blend)
                        else (tuple(np.ones(oof_n_classes, dtype=float).tolist()),)
                    ),
                    gate_thresholds=gate_thresholds if bool(args.use_blend) else (0.0,),
                    persist_target_metrics=oof_persist_target_metrics,
                    persist_regime_metrics=oof_persist_regime_metrics,
                    p_change=oof_p_change,
                    change_gate_thresholds=(
                        gkg_change_gate_thresholds if bool(args.use_gkg_change_gate) else (0.0,)
                    ),
                    change_alpha_weights=(
                        asset_gkg_change_alpha_weights
                        if bool(args.use_gkg_change_alpha)
                        else (0.0,)
                    ),
                    min_non_transition_delta=float(args.min_non_transition_delta),
                )
                oof_selected_alpha = float(oof_selected_cfg["selected_alpha"])
                oof_selected_beta = float(oof_selected_cfg["selected_beta"])
                oof_selected_class_thresholds = _parse_class_thresholds(
                    oof_selected_cfg["selected_class_thresholds"],
                    n_classes=oof_n_classes,
                )
                oof_selected_gate_threshold = float(
                    np.clip(oof_selected_cfg["selected_gate_threshold"], 0.0, 1.0)
                )
                oof_selected_change_gate_threshold = float(
                    np.clip(oof_selected_cfg.get("selected_change_gate_threshold", 0.0), 0.0, 1.0)
                )
                oof_selected_change_alpha_weight = float(
                    np.clip(oof_selected_cfg.get("selected_change_alpha_weight", 0.0), 0.0, 1.0)
                )

                fold_eval_metrics = []
                for r in rows:
                    fold_persist = {
                        "acc": float(r["persist_valid_acc"]),
                        "macro_f1": float(r["persist_valid_macro_f1"]),
                        "weighted_f1": float(r["persist_valid_weighted_f1"]),
                        "macro_recall": float(r["persist_valid_macro_recall"]),
                        "high_vol_recall": float(r["persist_valid_high_vol_recall"]),
                    }
                    fold_persist_target = {
                        "acc": float(r["persist_valid_target_acc"]),
                        "macro_f1": float(r["persist_valid_target_macro_f1"]),
                        "weighted_f1": float(r["persist_valid_target_weighted_f1"]),
                        "macro_recall": float(r["persist_valid_target_macro_recall"]),
                        "high_vol_recall": float(r["persist_valid_target_high_vol_recall"]),
                    }
                    fm = _blend_metrics(
                        proba_chain=np.asarray(r["proba_valid"], dtype=float),
                        y_true_target=np.asarray(r["y_valid"], dtype=int),
                        y_persist_target=np.asarray(r["persist_valid_pred"], dtype=int),
                        y_true_regime=np.asarray(r["y_valid_regime"], dtype=int),
                        persist_target_metrics=fold_persist_target,
                        persist_regime_metrics=fold_persist,
                        alpha=float(oof_selected_alpha),
                        beta=float(oof_selected_beta),
                        class_thresholds=oof_selected_class_thresholds,
                        gate_threshold=oof_selected_gate_threshold,
                        p_change=(
                            np.asarray(r["gkg_change_p_change_eval"], dtype=float)
                            if bool(args.use_gkg_change_gate or args.use_gkg_change_alpha)
                            else None
                        ),
                        change_gate_threshold=float(oof_selected_change_gate_threshold),
                        change_alpha_weight=float(oof_selected_change_alpha_weight),
                    )
                    fold_eval_metrics.append(fm)

                acc_series = np.array([m["acc"] for m in fold_eval_metrics], dtype=float)
                macro_series = np.array([m["macro_f1"] for m in fold_eval_metrics], dtype=float)
                weighted_series = np.array([m["weighted_f1"] for m in fold_eval_metrics], dtype=float)
                macro_recall_series = np.array([m["macro_recall"] for m in fold_eval_metrics], dtype=float)
                high_vol_recall_series = np.array(
                    [m["high_vol_recall"] for m in fold_eval_metrics], dtype=float
                )
                delta_series = np.array(
                    [m["delta_macro_f1_vs_persistence"] for m in fold_eval_metrics], dtype=float
                )
                delta_weighted_series = np.array(
                    [m["delta_weighted_f1_vs_persistence"] for m in fold_eval_metrics], dtype=float
                )
                delta_macro_recall_series = np.array(
                    [m["delta_macro_recall_vs_persistence"] for m in fold_eval_metrics], dtype=float
                )
                delta_high_vol_recall_series = np.array(
                    [m["delta_high_vol_recall_vs_persistence"] for m in fold_eval_metrics], dtype=float
                )
                delta_transition_macro_f1_series = np.array(
                    [m["delta_transition_macro_f1_vs_persistence"] for m in fold_eval_metrics],
                    dtype=float,
                )
                delta_non_transition_macro_f1_series = np.array(
                    [m["delta_non_transition_macro_f1_vs_persistence"] for m in fold_eval_metrics],
                    dtype=float,
                )
                gating_rate_series = np.array([m["gating_rate"] for m in fold_eval_metrics], dtype=float)
                confidence_gating_rate_series = np.array(
                    [m["confidence_gating_rate"] for m in fold_eval_metrics], dtype=float
                )
                change_gating_rate_series = np.array(
                    [m["change_gating_rate"] for m in fold_eval_metrics], dtype=float
                )
                mean_p_change_series = np.array(
                    [m["mean_p_change"] for m in fold_eval_metrics], dtype=float
                )
                confidence_series = np.array([m["mean_confidence"] for m in fold_eval_metrics], dtype=float)
                change_alpha_multiplier_series = np.array(
                    [m["mean_change_alpha_multiplier"] for m in fold_eval_metrics], dtype=float
                )
                gkg_change_enabled_series = np.array(
                    [float(bool(r.get("gkg_change_enabled", False))) for r in rows], dtype=float
                )
                gkg_change_macro_f1_series = np.array(
                    [float(r.get("gkg_change_eval_macro_f1", np.nan)) for r in rows], dtype=float
                )
                gkg_change_auc_series = np.array(
                    [float(r.get("gkg_change_eval_auc", np.nan)) for r in rows], dtype=float
                )
                gkg_change_brier_series = np.array(
                    [float(r.get("gkg_change_eval_brier", np.nan)) for r in rows], dtype=float
                )
                gkg_change_pchange_mean_series = np.array(
                    [float(r.get("gkg_change_eval_p_change_mean", np.nan)) for r in rows],
                    dtype=float,
                )

                positive_rate = float(np.mean(delta_series > 0.0))
                stability_penalty = float(args.stability_lambda) * float(np.std(delta_series))
                positive_gap = max(0.0, float(args.min_positive_rate) - positive_rate)
                positive_penalty = float(args.positive_rate_lambda) * positive_gap
                mean_non_transition_delta = float(
                    np.nan_to_num(_safe_mean(delta_non_transition_macro_f1_series), nan=-1.0)
                )
                non_transition_gap = max(
                    0.0, float(args.min_non_transition_delta) - mean_non_transition_delta
                )
                non_transition_penalty = (
                    float(args.non_transition_penalty_lambda) * non_transition_gap
                )
                transition_bonus = float(args.transition_bonus_lambda) * float(
                    np.nan_to_num(_safe_mean(delta_transition_macro_f1_series), nan=0.0)
                )
                robust_score = (
                    float(np.mean(delta_series))
                    - stability_penalty
                    - positive_penalty
                    - non_transition_penalty
                    + transition_bonus
                )

                gate_series_fold = np.array(
                    [float(r.get("selected_gate_threshold", 0.0)) for r in rows], dtype=float
                )
                change_gate_series_fold = np.array(
                    [float(r.get("selected_change_gate_threshold", 0.0)) for r in rows],
                    dtype=float,
                )
                change_alpha_weight_series_fold = np.array(
                    [float(r.get("selected_change_alpha_weight", 0.0)) for r in rows],
                    dtype=float,
                )
                class_threshold_tokens = [
                    str(r.get("selected_class_thresholds", _format_class_thresholds([1.0, 1.0, 1.0])))
                    for r in rows
                ]
                class_threshold_mode = Counter(class_threshold_tokens).most_common(1)[0][0]
                cal_method_tokens = [str(r.get("calibration_method", "none")) for r in rows]
                calibration_method_mode = Counter(cal_method_tokens).most_common(1)[0][0]

                summary_rows.append(
                    {
                        "asset": asset_name,
                        "tickers": "|".join(group_tickers),
                        "tabular_model": str(args.tabular_model),
                        "struct_id": s_idx,
                        "model_id": x_idx,
                        "xgb_id": x_idx,
                        **s_cfg,
                        **xgb_cfgs[x_idx],
                        "n_folds_ok": int(len(rows)),
                        "mean_selected_alpha": float(np.mean(alpha_series_fold)),
                        "mean_selected_beta": float(np.mean(beta_series_fold)),
                        "mean_selected_gate_threshold": float(np.mean(gate_series_fold)),
                        "mean_selected_change_gate_threshold": float(
                            np.mean(change_gate_series_fold)
                        ),
                        "mean_selected_change_alpha_weight": float(
                            np.mean(change_alpha_weight_series_fold)
                        ),
                        "mean_selected_class_thresholds": class_threshold_mode,
                        "selected_calibration_method": calibration_method_mode,
                        "mean_effective_alpha_fold": float(np.mean(eff_alpha_series_fold)),
                        "mean_change_alpha_multiplier_fold": _safe_mean(
                            change_alpha_multiplier_series
                        ),
                        "mean_confidence_fold": _safe_mean(confidence_series),
                        "mean_gating_rate_fold": _safe_mean(gating_rate_series),
                        "mean_confidence_gating_rate_fold": _safe_mean(
                            confidence_gating_rate_series
                        ),
                        "mean_change_gating_rate_fold": _safe_mean(change_gating_rate_series),
                        "mean_p_change_fold": _safe_mean(mean_p_change_series),
                        "gkg_change_enabled_rate_fold": _safe_mean(gkg_change_enabled_series),
                        "use_gkg_change_gate": bool(args.use_gkg_change_gate),
                        "use_gkg_change_alpha": bool(args.use_gkg_change_alpha),
                        "gkg_change_alpha_weights_grid": ",".join(
                            str(v) for v in asset_gkg_change_alpha_weights
                        ),
                        "gkg_change_model": str(
                            Counter([str(r.get("gkg_change_model", "")) for r in rows]).most_common(1)[0][0]
                        ),
                        "gkg_change_context_cols": str(
                            Counter([str(r.get("gkg_change_context_cols", "")) for r in rows]).most_common(1)[0][0]
                        ),
                        "gkg_change_calibration_method": str(
                            Counter(
                                [str(r.get("gkg_change_calibration_method", "")) for r in rows]
                            ).most_common(1)[0][0]
                        ),
                        "mean_gkg_change_n_features": _safe_mean(
                            [float(r.get("gkg_change_n_features", np.nan)) for r in rows]
                        ),
                        "mean_gkg_change_eval_macro_f1": _safe_mean(gkg_change_macro_f1_series),
                        "mean_gkg_change_eval_auc": _safe_mean(gkg_change_auc_series),
                        "mean_gkg_change_eval_brier": _safe_mean(gkg_change_brier_series),
                        "mean_gkg_change_eval_p_change_mean": _safe_mean(
                            gkg_change_pchange_mean_series
                        ),
                        "gkg_change_blend_hook_weight": float(
                            args.gkg_change_blend_hook_weight
                        ),
                        "oof_selected_alpha": float(oof_selected_alpha),
                        "oof_selected_beta": float(oof_selected_beta),
                        "oof_selected_gate_threshold": float(oof_selected_gate_threshold),
                        "oof_selected_change_gate_threshold": float(
                            oof_selected_change_gate_threshold
                        ),
                        "oof_selected_change_alpha_weight": float(
                            oof_selected_change_alpha_weight
                        ),
                        "oof_selected_class_thresholds": _format_class_thresholds(
                            oof_selected_class_thresholds
                        ),
                        "oof_mean_effective_alpha": float(oof_best_m["mean_effective_alpha"]),
                        "oof_mean_change_alpha_multiplier": float(
                            oof_best_m["mean_change_alpha_multiplier"]
                        ),
                        "oof_mean_confidence": float(oof_best_m["mean_confidence"]),
                        "oof_gating_rate": float(oof_best_m["gating_rate"]),
                        "oof_confidence_gating_rate": float(
                            oof_best_m["confidence_gating_rate"]
                        ),
                        "oof_change_gating_rate": float(oof_best_m["change_gating_rate"]),
                        "oof_mean_p_change": float(oof_best_m["mean_p_change"]),
                        "oof_valid_acc": float(oof_best_m["acc"]),
                        "oof_valid_macro_f1": float(oof_best_m["macro_f1"]),
                        "oof_valid_weighted_f1": float(oof_best_m["weighted_f1"]),
                        "oof_valid_macro_recall": float(oof_best_m["macro_recall"]),
                        "oof_valid_high_vol_recall": float(oof_best_m["high_vol_recall"]),
                        "oof_valid_target_acc": float(oof_best_m["target_acc"]),
                        "oof_valid_target_macro_f1": float(oof_best_m["target_macro_f1"]),
                        "oof_valid_target_weighted_f1": float(oof_best_m["target_weighted_f1"]),
                        "oof_valid_target_macro_recall": float(oof_best_m["target_macro_recall"]),
                        "oof_valid_target_high_vol_recall": float(
                            oof_best_m["target_high_vol_recall"]
                        ),
                        "oof_delta_macro_vs_persistence": float(
                            oof_best_m["delta_macro_f1_vs_persistence"]
                        ),
                        "oof_delta_transition_macro_f1_vs_persistence": float(
                            oof_best_m["delta_transition_macro_f1_vs_persistence"]
                        ),
                        "oof_delta_high_vol_recall_vs_persistence": float(
                            oof_best_m["delta_high_vol_recall_vs_persistence"]
                        ),
                        "mean_valid_acc": float(np.mean(acc_series)),
                        "mean_valid_macro_f1": float(np.mean(macro_series)),
                        "mean_valid_weighted_f1": float(np.mean(weighted_series)),
                        "mean_valid_macro_recall": float(np.mean(macro_recall_series)),
                        "mean_valid_high_vol_recall": float(np.mean(high_vol_recall_series)),
                        "std_valid_macro_f1": float(np.std(macro_series)),
                        "mean_delta_macro_vs_persistence": float(np.mean(delta_series)),
                        "mean_delta_weighted_vs_persistence": float(np.mean(delta_weighted_series)),
                        "mean_delta_macro_recall_vs_persistence": float(
                            np.mean(delta_macro_recall_series)
                        ),
                        "mean_delta_high_vol_recall_vs_persistence": float(
                            np.mean(delta_high_vol_recall_series)
                        ),
                        "mean_delta_transition_macro_f1_vs_persistence": _safe_mean(
                            delta_transition_macro_f1_series
                        ),
                        "mean_delta_non_transition_macro_f1_vs_persistence": _safe_mean(
                            delta_non_transition_macro_f1_series
                        ),
                        "std_delta_macro_vs_persistence": float(np.std(delta_series)),
                        "positive_delta_rate": positive_rate,
                        "non_transition_penalty": float(non_transition_penalty),
                        "transition_bonus": float(transition_bonus),
                        "robust_score": robust_score,
                        "pruned": bool(pruned_at[x_idx] is not None),
                        "pruned_at_fold": pruned_at[x_idx],
                    }
                )

        summary_df = pd.DataFrame(summary_rows)
        if summary_df.empty:
            failed_rows = [r for r in fold_rows if "error" in r]
            if failed_rows:
                pd.DataFrame(failed_rows).to_csv(outdir / "failed_experiments.csv", index=False)
            raise SystemExit(f"No successful experiments in walk-forward for asset={asset_name}.")

        summary_df = summary_df[summary_df["n_folds_ok"] >= int(args.min_folds_ok)].copy()
        if summary_df.empty:
            raise SystemExit(f"No experiments satisfied min_folds_ok for asset={asset_name}.")

        summary_df = summary_df[
            summary_df["mean_delta_high_vol_recall_vs_persistence"]
            >= float(args.min_high_vol_recall_delta)
        ].copy()
        if summary_df.empty:
            raise SystemExit(
                f"No experiments satisfied high-vol recall constraint for asset={asset_name}. "
                f"Try lowering --min_high_vol_recall_delta."
            )

        summary_df_non_transition = summary_df[
            summary_df["mean_delta_non_transition_macro_f1_vs_persistence"]
            >= float(args.min_non_transition_delta)
        ].copy()
        if not summary_df_non_transition.empty:
            summary_df = summary_df_non_transition

        summary_df = summary_df.sort_values(
            [
                "robust_score",
                "mean_delta_non_transition_macro_f1_vs_persistence",
                "mean_delta_high_vol_recall_vs_persistence",
                "mean_delta_macro_vs_persistence",
                "mean_valid_macro_f1",
            ],
            ascending=[False, False, False, False, False],
        )
        best = summary_df.iloc[0].to_dict()
        global_best_rows.append(best)

        pd.DataFrame(fold_rows).to_csv(outdir / "fold_metrics.csv", index=False)
        summary_df.to_csv(outdir / "summary.csv", index=False)
        with open(outdir / "best.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)

        final_cmp = _official_test_compare(
            db_path=db_path,
            tickers=group_tickers,
            horizon=int(args.horizon),
            regime_bins=regime_bins,
            train_end=features_cfg["split"]["train_end"],
            valid_end=features_cfg["split"]["val_end"],
            struct_cfg={
                "chain_mean_model": best.get("chain_mean_model", "sarimax"),
                "sarimax_order": best["sarimax_order"],
                "sarimax_chain_exog_cols": best["sarimax_chain_exog_cols"],
                "har_target_col": best.get("har_target_col", "rv_20"),
                "har_lag_1": best.get("har_lag_1", 1),
                "har_lag_week": best.get("har_lag_week", 5),
                "har_lag_month": best.get("har_lag_month", 22),
                "har_exog_cols": best.get("har_exog_cols", ()),
                "garch_p": best["garch_p"],
                "garch_o": best.get("garch_o", 1),
                "garch_q": best["garch_q"],
                "garch_dist": best["garch_dist"],
                "garch_vol": best.get("garch_vol", "Garch"),
                "garch_chain_agg": best["garch_chain_agg"],
                "garch_scale": best["garch_scale"],
            },
            model_cfg=(
                {
                    "n_estimators": best["n_estimators"],
                    "max_depth": best["max_depth"],
                    "learning_rate": best["learning_rate"],
                    "subsample": best["subsample"],
                    "colsample_bytree": best["colsample_bytree"],
                    "min_child_weight": best["min_child_weight"],
                    "reg_lambda": best["reg_lambda"],
                }
                if str(args.tabular_model) == "xgb"
                else {
                    "n_estimators": best.get("n_estimators", 8),
                    "softmax_temperature": best.get("softmax_temperature", 0.9),
                    "balance_probabilities": best.get("balance_probabilities", False),
                    "average_before_softmax": best.get("average_before_softmax", False),
                    "model_path": best.get(
                        "model_path", "tabpfn-v2-classifier-v2_default.ckpt"
                    ),
                    "device": best.get("device", str(args.tabpfn_device)),
                    "ignore_pretraining_limits": best.get("ignore_pretraining_limits", True),
                    "inference_precision": best.get("inference_precision", "auto"),
                    "fit_mode": best.get("fit_mode", "fit_preprocessors"),
                    "memory_saving_mode": best.get("memory_saving_mode", "auto"),
                    "random_state": best.get("random_state", 42),
                    "n_preprocessing_jobs": best.get(
                        "n_preprocessing_jobs", int(args.tabpfn_n_preprocessing_jobs)
                    ),
                }
            ),
            tabular_model=str(args.tabular_model),
            xgb_n_jobs=int(args.xgb_n_jobs),
            use_blend=bool(args.use_blend),
            blend_alpha=float(best.get("oof_selected_alpha", best.get("mean_selected_alpha", 1.0))),
            blend_beta=float(best.get("oof_selected_beta", 0.0)),
            blend_class_thresholds=_parse_class_thresholds(
                best.get(
                    "oof_selected_class_thresholds",
                    best.get("mean_selected_class_thresholds", "1|1|1"),
                ),
                n_classes=3,
            ),
            blend_gate_threshold=float(
                best.get(
                    "oof_selected_gate_threshold",
                    best.get("mean_selected_gate_threshold", 0.0),
                )
            ),
            calibration_method=str(best.get("selected_calibration_method", args.calibration_method)),
            use_gkg_change_detector=bool(args.use_gkg_change_detector),
            use_gkg_change_gate=bool(args.use_gkg_change_gate),
            use_gkg_change_alpha=bool(args.use_gkg_change_alpha),
            gkg_change_model=str(args.gkg_change_model),
            gkg_change_context_cols=gkg_change_context_cols,
            gkg_change_calibration_method=str(args.gkg_change_calibration_method),
            gkg_change_gate_threshold=float(
                best.get(
                    "oof_selected_change_gate_threshold",
                    best.get("mean_selected_change_gate_threshold", 0.0),
                )
            ),
            gkg_change_alpha_weight=float(
                best.get(
                    "oof_selected_change_alpha_weight",
                    best.get("mean_selected_change_alpha_weight", 0.0),
                )
            ),
            gkg_change_blend_hook_weight=float(args.gkg_change_blend_hook_weight),
        )
        final_compare_rows.append(
            {
                "asset": asset_name,
                "tickers": "|".join(group_tickers),
                "horizon": int(args.horizon),
                "tabular_model": str(args.tabular_model),
                "selected_struct_id": int(best["struct_id"]),
                "selected_model_id": int(best["xgb_id"]),
                "selected_xgb_id": int(best["xgb_id"]),
                **final_cmp,
            }
        )
        pd.DataFrame([final_compare_rows[-1]]).to_csv(
            outdir / "final_vs_persistence.csv", index=False
        )

        print(
            f"Asset={asset_name} | Folds={len(folds)} | Struct={len(structural_cfgs)} | "
            f"{str(args.tabular_model).upper()}={len(xgb_cfgs)}"
        )
        print(summary_df.head(5).to_string(index=False))
        print("\nSaved:", outdir)

    root_out = Path(args.outdir) / f"h{args.horizon}"
    if global_best_rows:
        pd.DataFrame(global_best_rows).to_csv(root_out / "best_by_asset.csv", index=False)
        print("\nBest-by-asset:", root_out / "best_by_asset.csv")
    if final_compare_rows and len(groups) > 1:
        pd.DataFrame(final_compare_rows).to_csv(root_out / "final_vs_persistence.csv", index=False)
        print("Final-vs-persistence:", root_out / "final_vs_persistence.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
