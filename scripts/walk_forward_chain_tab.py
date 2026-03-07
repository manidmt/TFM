'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-28

@description: Walk-forward model selection for chained SARIMAX->GARCH->tabular
using robust delta macro F1 vs persistence baseline.
'''

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_risk.datasets.make_dataset import DatasetConfig, make_dataset
from quant_risk.models.baseline import persistence_pred_from_regime
from quant_risk.models.metrics import compute_metrics
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

DEFAULT_CHAIN_VARIANTS_CONFIG = "src/quant_risk/models/econometric/chain_variants.yaml"
DEFAULT_TABPFN_VARIANTS_CONFIG = "src/quant_risk/models/tabular/tabpfn_variants.yaml"


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp("M")


def _normalize_order(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) == 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
    raise ValueError(f"Invalid sarimax_order={value!r}; expected [p,d,q] or 'p,d,q'.")


def _normalize_exog_cols(value: Any) -> tuple[str, ...]:
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
    return {
        "sarimax_order": _normalize_order(row["sarimax_order"]),
        "sarimax_chain_exog_cols": _normalize_exog_cols(
            row.get("sarimax_chain_exog_cols", ())
        ),
        "garch_p": int(row["garch_p"]),
        "garch_o": int(row.get("garch_o", 1)),
        "garch_q": int(row["garch_q"]),
        "garch_dist": str(row["garch_dist"]),
        "garch_vol": str(row.get("garch_vol", "Garch")),
        "garch_chain_agg": str(row["garch_chain_agg"]),
        "garch_scale": float(row["garch_scale"]),
    }


def _load_chain_variants_from_yaml(profile: str, variants_path: str) -> list[dict[str, Any]]:
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
    orders = [_normalize_order(v) for v in axes.get("sarimax_order", [])]
    exogs = [_normalize_exog_cols(v) for v in axes.get("sarimax_chain_exog_cols", [()])]
    p_vals = [int(v) for v in axes.get("garch_p", [])]
    o_vals = [int(v) for v in axes.get("garch_o", [1])]
    q_vals = [int(v) for v in axes.get("garch_q", [])]
    dists = [str(v) for v in axes.get("garch_dist", [])]
    vols = [str(v) for v in axes.get("garch_vol", ["Garch"])]
    aggs = [str(v) for v in axes.get("garch_chain_agg", [])]
    scales = [float(v) for v in axes.get("garch_scale", [])]
    if not all([orders, exogs, p_vals, o_vals, q_vals, dists, vols, aggs, scales]):
        raise ValueError(
            f"{variants_path}: profiles.full_axes is missing required non-empty keys."
        )
    out: list[dict[str, Any]] = []
    for order, exog, p, o, q, dist, vol, agg, scale in itertools.product(
        orders, exogs, p_vals, o_vals, q_vals, dists, vols, aggs, scales
    ):
        out.append(
            {
                "sarimax_order": order,
                "sarimax_chain_exog_cols": exog,
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
    yaml_cfgs = _load_chain_variants_from_yaml(profile, variants_path)
    if yaml_cfgs:
        return yaml_cfgs
    if profile == "promising":
        return [
            {
                "sarimax_order": (2, 0, 1),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "garch_p": 2,
                "garch_o": 1,
                "garch_q": 2,
                "garch_dist": "skewt",
                "garch_vol": "Garch",
                "garch_chain_agg": "rms",
                "garch_scale": 100.0,
            },
            {
                "sarimax_order": (2, 0, 1),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "garch_p": 2,
                "garch_o": 1,
                "garch_q": 2,
                "garch_dist": "tstudent",
                "garch_vol": "Garch",
                "garch_chain_agg": "rms",
                "garch_scale": 100.0,
            },
            {
                "sarimax_order": (2, 0, 1),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "garch_p": 1,
                "garch_o": 1,
                "garch_q": 2,
                "garch_dist": "tstudent",
                "garch_vol": "Garch",
                "garch_chain_agg": "mean",
                "garch_scale": 80.0,
            },
            {
                "sarimax_order": (1, 0, 1),
                "sarimax_chain_exog_cols": (),
                "garch_p": 1,
                "garch_o": 1,
                "garch_q": 1,
                "garch_dist": "tstudent",
                "garch_vol": "Garch",
                "garch_chain_agg": "rms",
                "garch_scale": 100.0,
            },
            {
                "sarimax_order": (1, 0, 0),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "garch_p": 1,
                "garch_o": 1,
                "garch_q": 1,
                "garch_dist": "tstudent",
                "garch_vol": "Garch",
                "garch_chain_agg": "last",
                "garch_scale": 80.0,
            },
            {
                "sarimax_order": (2, 0, 1),
                "sarimax_chain_exog_cols": (),
                "garch_p": 2,
                "garch_o": 1,
                "garch_q": 1,
                "garch_dist": "tstudent",
                "garch_vol": "Garch",
                "garch_chain_agg": "rms",
                "garch_scale": 100.0,
            },
            {
                "sarimax_order": (1, 0, 1),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "garch_p": 2,
                "garch_o": 1,
                "garch_q": 1,
                "garch_dist": "tstudent",
                "garch_vol": "Garch",
                "garch_chain_agg": "mean",
                "garch_scale": 100.0,
            },
        ]

    orders = [(1, 0, 1), (2, 0, 1)]
    exogs = [(), ("net_liquidity_diff",)]
    p_vals = [1, 2]
    o_vals = [1]
    q_vals = [1, 2]
    dists = ["tstudent", "skewt"]
    vols = ["Garch"]
    aggs = ["rms", "mean"]
    scales = [80.0, 100.0]

    out = []
    for order, exog, p, o, q, dist, vol, agg, scale in itertools.product(
        orders, exogs, p_vals, o_vals, q_vals, dists, vols, aggs, scales
    ):
        out.append(
            {
                "sarimax_order": order,
                "sarimax_chain_exog_cols": exog,
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
    if not getattr(metrics_obj, "class_recall", None):
        return float("nan")
    high_vol_label = max(metrics_obj.class_recall.keys())
    return float(metrics_obj.class_recall.get(high_vol_label, float("nan")))


def _one_hot_probs(y: np.ndarray, n_classes: int) -> np.ndarray:
    y_arr = np.asarray(y, dtype=int)
    out = np.zeros((len(y_arr), int(n_classes)), dtype=float)
    if len(y_arr):
        out[np.arange(len(y_arr)), y_arr] = 1.0
    return out


def _parse_unit_interval_list(spec: str, *, name: str) -> tuple[float, ...]:
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
    return _parse_unit_interval_list(spec, name="blend_alphas")


def _parse_blend_conf_betas(spec: str) -> tuple[float, ...]:
    return _parse_unit_interval_list(spec, name="blend_conf_betas")


def _aligned_split_with_persistence(
    df_all: pd.DataFrame,
    split_df: pd.DataFrame,
    horizon: int,
) -> dict[str, Any]:
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
    alpha_c = float(np.clip(alpha, 0.0, 1.0))
    beta_c = float(np.clip(beta, 0.0, 1.0))
    persist_proba = _one_hot_probs(y_persist, n_classes=int(proba_chain.shape[1]))

    if beta_c <= 0.0:
        alpha_vec = np.full(len(proba_chain), alpha_c, dtype=float)
    else:
        # Confidence proxy from chain probabilities: max class probability.
        conf = np.max(proba_chain, axis=1).astype(float)
        alpha_vec = np.clip((1.0 - beta_c) * alpha_c + beta_c * conf, 0.0, 1.0)

    p_mix = alpha_vec[:, None] * proba_chain + (1.0 - alpha_vec[:, None]) * persist_proba
    pred = np.argmax(p_mix, axis=1).astype(int)
    return pred, alpha_vec


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
) -> dict[str, float]:
    pred_target, alpha_vec = _blend_predict(
        proba_chain=proba_chain,
        y_persist=y_persist_target,
        alpha=alpha,
        beta=beta,
    )

    m_target = compute_metrics(y_true_target, pred_target)
    target_high_vol_recall = _high_vol_recall(m_target)

    pred_regime = np.asarray(pred_target, dtype=int)

    m_regime = compute_metrics(y_true_regime, pred_regime)
    regime_high_vol_recall = _high_vol_recall(m_regime)

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
        "pred_target": np.asarray(pred_target, dtype=int),
        "pred_regime": np.asarray(pred_regime, dtype=int),
    }


def _pick_best_blend_alpha(
    proba_chain: np.ndarray,
    y_true_target: np.ndarray,
    y_persist_target: np.ndarray,
    y_true_regime: np.ndarray,
    blend_alphas: tuple[float, ...],
    persist_target_metrics: dict[str, float],
    persist_regime_metrics: dict[str, float],
) -> tuple[float, dict[str, float]]:
    best_alpha = float(blend_alphas[-1])
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float] | None = None
    for alpha in blend_alphas:
        cand = _blend_metrics(
            proba_chain=proba_chain,
            y_true_target=y_true_target,
            y_persist_target=y_persist_target,
            y_true_regime=y_true_regime,
            persist_target_metrics=persist_target_metrics,
            persist_regime_metrics=persist_regime_metrics,
            alpha=float(alpha),
            beta=0.0,
        )
        score = (
            cand["delta_macro_f1_vs_persistence"],
            cand["delta_high_vol_recall_vs_persistence"],
            cand["acc"],
        )
        if best_score is None or score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_metrics = cand
    assert best_metrics is not None
    return best_alpha, best_metrics


def _pick_best_blend_beta(
    proba_chain: np.ndarray,
    y_true_target: np.ndarray,
    y_persist_target: np.ndarray,
    y_true_regime: np.ndarray,
    alpha: float,
    blend_conf_betas: tuple[float, ...],
    persist_target_metrics: dict[str, float],
    persist_regime_metrics: dict[str, float],
) -> tuple[float, dict[str, float]]:
    best_beta = float(blend_conf_betas[0])
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float] | None = None
    for beta in blend_conf_betas:
        cand = _blend_metrics(
            proba_chain=proba_chain,
            y_true_target=y_true_target,
            y_persist_target=y_persist_target,
            y_true_regime=y_true_regime,
            persist_target_metrics=persist_target_metrics,
            persist_regime_metrics=persist_regime_metrics,
            alpha=float(alpha),
            beta=float(beta),
        )
        score = (
            cand["delta_macro_f1_vs_persistence"],
            cand["delta_high_vol_recall_vs_persistence"],
            cand["acc"],
        )
        if best_score is None or score > best_score:
            best_score = score
            best_beta = float(beta)
            best_metrics = cand
    assert best_metrics is not None
    return best_beta, best_metrics


def _cache_key(
    asset: str,
    horizon: int,
    train_end: str,
    valid_end: str,
    struct_cfg: dict[str, Any],
) -> str:
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
    asset_name = "_".join(tickers)
    ck = _cache_key(asset_name, horizon, train_end, valid_end, struct_cfg)
    cp = cache_dir / f"fold_{ck}.npz"
    required_cache_keys = {
        "x_train",
        "y_train",
        "x_valid",
        "y_valid",
        "y_valid_regime",
        "regime_prev_valid",
        "persist_valid_pred",
        "persist_valid_regime_pred",
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
                return float(np.asarray(z[name]).reshape(-1)[0])

            return {
                "x_train": z["x_train"],
                "y_train": z["y_train"],
                "x_valid": z["x_valid"],
                "y_valid": z["y_valid"],
                "y_valid_regime": z["y_valid_regime"],
                "regime_prev_valid": z["regime_prev_valid"],
                "persist_valid_pred": np.asarray(z["persist_valid_pred"], dtype=int),
                "persist_valid_regime_pred": np.asarray(z["persist_valid_regime_pred"], dtype=int),
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
        sarimax_order=tuple(struct_cfg["sarimax_order"]),
        sarimax_chain_exog_cols=tuple(struct_cfg["sarimax_chain_exog_cols"]),
        garch_p=int(struct_cfg["garch_p"]),
        garch_o=int(struct_cfg.get("garch_o", 1)),
        garch_q=int(struct_cfg["garch_q"]),
        garch_dist=str(struct_cfg["garch_dist"]),
        garch_vol=str(struct_cfg.get("garch_vol", "Garch")),
        garch_chain_agg=str(struct_cfg["garch_chain_agg"]),
        garch_scale=float(struct_cfg["garch_scale"]),
    )
    pack = make_dataset(dcfg)

    x_train = pack["train"][pack["feature_cols"]].to_numpy(dtype=np.float32)
    y_train = pack["train"]["regime"].to_numpy(dtype=int)
    x_valid_full = pack["valid"][pack["feature_cols"]].to_numpy(dtype=np.float32)
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
        "x_valid": x_valid,
        "y_valid": y_valid,
        "y_valid_regime": y_valid_regime,
        "regime_prev_valid": regime_prev_valid,
        "persist_valid_pred": persist_valid_pred,
        "persist_valid_regime_pred": persist_valid_regime_pred,
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
            x_valid=x_valid,
            y_valid=y_valid,
            y_valid_regime=y_valid_regime,
            regime_prev_valid=regime_prev_valid,
            persist_valid_pred=persist_valid_pred,
            persist_valid_regime_pred=persist_valid_regime_pred,
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
) -> dict[str, Any]:
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
        proba_valid = predict_proba_tabpfn(model, fold_data["x_valid"])
    else:
        raise ValueError(f"Invalid tabular_model={tabular_model!r}")
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
    if bool(use_blend):
        selected_alpha, best_m = _pick_best_blend_alpha(
            proba_chain=proba_valid,
            y_true_target=fold_data["y_valid"],
            y_persist_target=fold_data["persist_valid_pred"],
            y_true_regime=fold_data["y_valid_regime"],
            blend_alphas=blend_alphas,
            persist_target_metrics=persist_target_metrics,
            persist_regime_metrics=persist_regime_metrics,
        )
        selected_beta, best_m = _pick_best_blend_beta(
            proba_chain=proba_valid,
            y_true_target=fold_data["y_valid"],
            y_persist_target=fold_data["persist_valid_pred"],
            y_true_regime=fold_data["y_valid_regime"],
            alpha=float(selected_alpha),
            blend_conf_betas=blend_conf_betas,
            persist_target_metrics=persist_target_metrics,
            persist_regime_metrics=persist_regime_metrics,
        )
    else:
        selected_alpha = 1.0
        selected_beta = 0.0
        best_m = _blend_metrics(
            proba_chain=proba_valid,
            y_true_target=fold_data["y_valid"],
            y_persist_target=fold_data["persist_valid_pred"],
            y_true_regime=fold_data["y_valid_regime"],
            persist_target_metrics=persist_target_metrics,
            persist_regime_metrics=persist_regime_metrics,
            alpha=1.0,
            beta=0.0,
        )

    return {
        "selected_alpha": float(selected_alpha),
        "selected_beta": float(selected_beta),
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
        "proba_valid": proba_valid,
        "y_valid": np.asarray(fold_data["y_valid"], dtype=int),
        "y_valid_regime": np.asarray(fold_data["y_valid_regime"], dtype=int),
        "regime_prev_valid": np.asarray(fold_data["regime_prev_valid"], dtype=int),
        "persist_valid_pred": np.asarray(fold_data["persist_valid_pred"], dtype=int),
        "persist_valid_regime_pred": np.asarray(
            fold_data["persist_valid_regime_pred"], dtype=int
        ),
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
) -> dict[str, Any]:
    dcfg = DatasetConfig(
        db_path=db_path,
        tickers=tickers,
        horizon=int(horizon),
        pooled=False,
        train_end=train_end,
        valid_end=valid_end,
        regime_bins=int(regime_bins),
        use_sarimax_garch_chain=True,
        sarimax_order=tuple(struct_cfg["sarimax_order"]),
        sarimax_chain_exog_cols=tuple(struct_cfg["sarimax_chain_exog_cols"]),
        garch_p=int(struct_cfg["garch_p"]),
        garch_o=int(struct_cfg.get("garch_o", 1)),
        garch_q=int(struct_cfg["garch_q"]),
        garch_dist=str(struct_cfg["garch_dist"]),
        garch_vol=str(struct_cfg.get("garch_vol", "Garch")),
        garch_chain_agg=str(struct_cfg["garch_chain_agg"]),
        garch_scale=float(struct_cfg["garch_scale"]),
    )
    pack = make_dataset(dcfg)

    x_train = pack["train"][pack["feature_cols"]].to_numpy(dtype=np.float32)
    y_train = pack["train"]["regime"].to_numpy(dtype=int)
    x_valid = pack["valid"][pack["feature_cols"]].to_numpy(dtype=np.float32)
    y_valid = pack["valid"]["regime"].to_numpy(dtype=int)
    x_test = pack["test"][pack["feature_cols"]].to_numpy(dtype=np.float32)
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
    else:
        raise ValueError(f"Invalid tabular_model={tabular_model!r}")

    aligned_test = _aligned_split_with_persistence(
        pack["df"],
        pack["test"],
        int(horizon),
    )
    keep_test = np.asarray(aligned_test["keep_mask"], dtype=bool)
    x_test_eval = x_test[keep_test]
    y_test_target_eval = y_test_target[keep_test]
    y_test_regime = np.asarray(aligned_test["y_regime_true"], dtype=int)
    regime_prev_test = np.asarray(aligned_test["regime_prev"], dtype=int)
    y_persist_test_target = np.asarray(aligned_test["y_target_persist"], dtype=int)
    p_metrics_regime = aligned_test["regime_persistence_metrics"]
    p_metrics_target = aligned_test["target_persistence_metrics"]

    if model_name == "xgb":
        proba_test = predict_proba_xgb(model, x_test_eval)
    else:
        proba_test = predict_proba_tabpfn(model, x_test_eval)
    if bool(use_blend):
        alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        beta = float(np.clip(blend_beta, 0.0, 1.0))
        pred_test, alpha_vec = _blend_predict(
            proba_chain=proba_test,
            y_persist=y_persist_test_target,
            alpha=alpha,
            beta=beta,
        )
    else:
        alpha = 1.0
        beta = 0.0
        pred_test, alpha_vec = _blend_predict(
            proba_chain=proba_test,
            y_persist=y_persist_test_target,
            alpha=alpha,
            beta=beta,
        )

    m_chain_target = compute_metrics(y_test_target_eval, pred_test)
    chain_target_high_vol_recall = _high_vol_recall(m_chain_target)
    pred_test_regime = np.asarray(pred_test, dtype=int)
    m_chain_regime = compute_metrics(y_test_regime, pred_test_regime)
    chain_regime_high_vol_recall = _high_vol_recall(m_chain_regime)

    return {
        "selected_blend_alpha": alpha,
        "selected_blend_beta": beta,
        "mean_effective_alpha_test": float(np.mean(alpha_vec) if len(alpha_vec) else alpha),
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
        "n_test_eval": int(len(y_test_regime)),
        "n_features": int(len(pack["feature_cols"])),
    }


def main() -> int:
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
    parser.add_argument("--disable_cache", action="store_true")
    parser.add_argument("--per_ticker", action="store_true")
    parser.add_argument(
        "--grid_profile",
        default="promising",
        choices=["promising", "full"],
        help="Hyperparameter grid profile: compact promising set or broader full set.",
    )
    args = parser.parse_args()
    blend_alphas = _parse_blend_alphas(args.blend_alphas)
    blend_conf_betas = _parse_blend_conf_betas(args.blend_conf_betas)

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
                        mean_high_vol_recall_delta = float(
                            np.mean([r["delta_high_vol_recall_vs_persistence"] for r in rows])
                        )
                        if (
                            mean_delta < float(args.prune_delta_threshold)
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

                if bool(args.use_blend):
                    oof_selected_alpha, _ = _pick_best_blend_alpha(
                        proba_chain=oof_proba,
                        y_true_target=oof_y_target,
                        y_persist_target=oof_persist_target,
                        y_true_regime=oof_y_regime,
                        blend_alphas=blend_alphas,
                        persist_target_metrics=oof_persist_target_metrics,
                        persist_regime_metrics=oof_persist_regime_metrics,
                    )
                    oof_selected_beta, oof_best_m = _pick_best_blend_beta(
                        proba_chain=oof_proba,
                        y_true_target=oof_y_target,
                        y_persist_target=oof_persist_target,
                        y_true_regime=oof_y_regime,
                        alpha=float(oof_selected_alpha),
                        blend_conf_betas=blend_conf_betas,
                        persist_target_metrics=oof_persist_target_metrics,
                        persist_regime_metrics=oof_persist_regime_metrics,
                    )
                else:
                    oof_selected_alpha = 1.0
                    oof_selected_beta = 0.0
                    oof_best_m = _blend_metrics(
                        proba_chain=oof_proba,
                        y_true_target=oof_y_target,
                        y_persist_target=oof_persist_target,
                        y_true_regime=oof_y_regime,
                        persist_target_metrics=oof_persist_target_metrics,
                        persist_regime_metrics=oof_persist_regime_metrics,
                        alpha=1.0,
                        beta=0.0,
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

                positive_rate = float(np.mean(delta_series > 0.0))
                stability_penalty = float(args.stability_lambda) * float(np.std(delta_series))
                positive_gap = max(0.0, float(args.min_positive_rate) - positive_rate)
                positive_penalty = float(args.positive_rate_lambda) * positive_gap
                robust_score = float(np.mean(delta_series)) - stability_penalty - positive_penalty

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
                        "mean_effective_alpha_fold": float(np.mean(eff_alpha_series_fold)),
                        "oof_selected_alpha": float(oof_selected_alpha),
                        "oof_selected_beta": float(oof_selected_beta),
                        "oof_mean_effective_alpha": float(oof_best_m["mean_effective_alpha"]),
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
                        "std_delta_macro_vs_persistence": float(np.std(delta_series)),
                        "positive_delta_rate": positive_rate,
                        "robust_score": robust_score,
                        "pruned": bool(pruned_at[x_idx] is not None),
                        "pruned_at_fold": pruned_at[x_idx],
                    }
                )

        summary_df = pd.DataFrame(summary_rows)
        if summary_df.empty:
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

        summary_df = summary_df.sort_values(
            [
                "robust_score",
                "mean_delta_high_vol_recall_vs_persistence",
                "mean_delta_macro_vs_persistence",
                "mean_valid_macro_f1",
            ],
            ascending=[False, False, False, False],
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
                "sarimax_order": best["sarimax_order"],
                "sarimax_chain_exog_cols": best["sarimax_chain_exog_cols"],
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
