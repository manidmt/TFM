'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-28

@description: Walk-forward model selection for chained SARIMAX->GARCH->XGB
using robust delta macro F1 vs persistence baseline.
'''

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_risk.datasets.make_dataset import DatasetConfig, make_dataset
from quant_risk.models.baseline import persistence_pred_from_regime
from quant_risk.models.metrics import compute_metrics
from quant_risk.models.tabular.xgb import (
    XGBConfig,
    fit as fit_xgb,
    make_model as make_xgb_model,
    predict_proba as predict_proba_xgb,
)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp("M")


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


def make_structural_configs(profile: str) -> list[dict[str, Any]]:
    if profile == "promising":
        return [
            {
                "sarimax_order": (2, 0, 1),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "garch_p": 2,
                "garch_q": 2,
                "garch_dist": "tstudent",
                "garch_chain_agg": "rms",
                "garch_scale": 100.0,
            },
            {
                "sarimax_order": (2, 0, 1),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "garch_p": 1,
                "garch_q": 2,
                "garch_dist": "tstudent",
                "garch_chain_agg": "mean",
                "garch_scale": 80.0,
            },
            {
                "sarimax_order": (1, 0, 1),
                "sarimax_chain_exog_cols": (),
                "garch_p": 1,
                "garch_q": 1,
                "garch_dist": "tstudent",
                "garch_chain_agg": "rms",
                "garch_scale": 100.0,
            },
            {
                "sarimax_order": (1, 0, 0),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "garch_p": 1,
                "garch_q": 1,
                "garch_dist": "tstudent",
                "garch_chain_agg": "last",
                "garch_scale": 80.0,
            },
            {
                "sarimax_order": (2, 0, 1),
                "sarimax_chain_exog_cols": (),
                "garch_p": 2,
                "garch_q": 1,
                "garch_dist": "tstudent",
                "garch_chain_agg": "rms",
                "garch_scale": 100.0,
            },
            {
                "sarimax_order": (1, 0, 1),
                "sarimax_chain_exog_cols": ("net_liquidity_diff",),
                "garch_p": 2,
                "garch_q": 1,
                "garch_dist": "tstudent",
                "garch_chain_agg": "mean",
                "garch_scale": 100.0,
            },
        ]

    orders = [(1, 0, 1), (2, 0, 1)]
    exogs = [(), ("net_liquidity_diff",)]
    p_vals = [1, 2]
    q_vals = [1, 2]
    dists = ["tstudent"]
    aggs = ["rms", "mean"]
    scales = [80.0, 100.0]

    out = []
    for order, exog, p, q, dist, agg, scale in itertools.product(
        orders, exogs, p_vals, q_vals, dists, aggs, scales
    ):
        out.append(
            {
                "sarimax_order": order,
                "sarimax_chain_exog_cols": exog,
                "garch_p": int(p),
                "garch_q": int(q),
                "garch_dist": str(dist),
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


def _parse_blend_alphas(spec: str) -> tuple[float, ...]:
    vals: list[float] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"blend alpha must be in [0,1], got {v}")
        vals.append(v)
    if not vals:
        raise ValueError("blend_alphas is empty")
    return tuple(sorted(set(vals)))


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
        "y_true": y_true,
        "y_persist": y_persist,
        "metrics": {
            "acc": float(m.accuracy),
            "macro_f1": float(m.macro_f1),
            "weighted_f1": float(m.weighted_f1),
            "macro_recall": float(m.macro_recall),
            "high_vol_recall": _high_vol_recall(m),
        },
    }


def _persist_metrics(df_all: pd.DataFrame, split_df: pd.DataFrame, horizon: int) -> dict[str, float]:
    aligned = _aligned_split_with_persistence(df_all, split_df, horizon)
    return aligned["metrics"]


def _pick_best_blend_alpha(
    proba_chain: np.ndarray,
    y_true: np.ndarray,
    y_persist: np.ndarray,
    blend_alphas: tuple[float, ...],
    persist_metrics: dict[str, float],
) -> tuple[float, dict[str, float]]:
    n_classes = int(proba_chain.shape[1])
    persist_proba = _one_hot_probs(y_persist, n_classes=n_classes)

    best_alpha = float(blend_alphas[-1])
    best_metrics: dict[str, float] | None = None
    best_score: tuple[float, float, float] | None = None
    for alpha in blend_alphas:
        p_mix = float(alpha) * proba_chain + (1.0 - float(alpha)) * persist_proba
        pred = np.argmax(p_mix, axis=1).astype(int)
        m = compute_metrics(y_true, pred)
        high_vol_recall = _high_vol_recall(m)
        cand = {
            "acc": float(m.accuracy),
            "macro_f1": float(m.macro_f1),
            "weighted_f1": float(m.weighted_f1),
            "macro_recall": float(m.macro_recall),
            "high_vol_recall": float(high_vol_recall),
            "delta_acc_vs_persistence": float(m.accuracy - persist_metrics["acc"]),
            "delta_macro_f1_vs_persistence": float(m.macro_f1 - persist_metrics["macro_f1"]),
            "delta_weighted_f1_vs_persistence": float(
                m.weighted_f1 - persist_metrics["weighted_f1"]
            ),
            "delta_macro_recall_vs_persistence": float(
                m.macro_recall - persist_metrics["macro_recall"]
            ),
            "delta_high_vol_recall_vs_persistence": float(
                high_vol_recall - persist_metrics["high_vol_recall"]
            ),
        }
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


def _cache_key(asset: str, horizon: int, train_end: str, valid_end: str, struct_cfg: dict[str, Any]) -> str:
    payload = {
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
        "persist_valid_pred",
        "persist_valid_acc",
        "persist_valid_macro_f1",
        "persist_valid_weighted_f1",
        "persist_valid_macro_recall",
        "persist_valid_high_vol_recall",
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
                "persist_valid_pred": np.asarray(z["persist_valid_pred"], dtype=int),
                "persist_valid_acc": _scalar("persist_valid_acc"),
                "persist_valid_macro_f1": _scalar("persist_valid_macro_f1"),
                "persist_valid_weighted_f1": _scalar("persist_valid_weighted_f1"),
                "persist_valid_macro_recall": _scalar("persist_valid_macro_recall"),
                "persist_valid_high_vol_recall": _scalar("persist_valid_high_vol_recall"),
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
        garch_q=int(struct_cfg["garch_q"]),
        garch_dist=str(struct_cfg["garch_dist"]),
        garch_chain_agg=str(struct_cfg["garch_chain_agg"]),
        garch_scale=float(struct_cfg["garch_scale"]),
    )
    pack = make_dataset(dcfg)

    x_train = pack["train"][pack["feature_cols"]].to_numpy(dtype=np.float32)
    y_train = pack["train"]["regime"].to_numpy(dtype=int)
    x_valid_full = pack["valid"][pack["feature_cols"]].to_numpy(dtype=np.float32)
    y_valid_full = pack["valid"]["regime"].to_numpy(dtype=int)
    aligned_valid = _aligned_split_with_persistence(pack["df"], pack["valid"], horizon=int(horizon))
    keep_valid = np.asarray(aligned_valid["keep_mask"], dtype=bool)
    x_valid = x_valid_full[keep_valid]
    y_valid = y_valid_full[keep_valid]
    persist_valid_pred = np.asarray(aligned_valid["y_persist"], dtype=int)
    p_metrics = aligned_valid["metrics"]

    out = {
        "x_train": x_train,
        "y_train": y_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "persist_valid_pred": persist_valid_pred,
        "persist_valid_acc": float(p_metrics["acc"]),
        "persist_valid_macro_f1": float(p_metrics["macro_f1"]),
        "persist_valid_weighted_f1": float(p_metrics["weighted_f1"]),
        "persist_valid_macro_recall": float(p_metrics["macro_recall"]),
        "persist_valid_high_vol_recall": float(p_metrics["high_vol_recall"]),
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
            persist_valid_pred=persist_valid_pred,
            persist_valid_acc=np.array([p_metrics["acc"]], dtype=float),
            persist_valid_macro_f1=np.array([p_metrics["macro_f1"]], dtype=float),
            persist_valid_weighted_f1=np.array([p_metrics["weighted_f1"]], dtype=float),
            persist_valid_macro_recall=np.array([p_metrics["macro_recall"]], dtype=float),
            persist_valid_high_vol_recall=np.array([p_metrics["high_vol_recall"]], dtype=float),
            n_features=np.array([out["n_features"]], dtype=int),
        )

    return out


def _fit_eval_xgb(
    xgb_cfg: dict[str, Any],
    fold_data: dict[str, Any],
    xgb_n_jobs: int,
    use_blend: bool,
    blend_alphas: tuple[float, ...],
) -> dict[str, Any]:
    cfg = XGBConfig(
        n_estimators=int(xgb_cfg["n_estimators"]),
        max_depth=int(xgb_cfg["max_depth"]),
        learning_rate=float(xgb_cfg["learning_rate"]),
        subsample=float(xgb_cfg["subsample"]),
        colsample_bytree=float(xgb_cfg["colsample_bytree"]),
        min_child_weight=float(xgb_cfg["min_child_weight"]),
        reg_lambda=float(xgb_cfg["reg_lambda"]),
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
    if bool(use_blend):
        selected_alpha, best_m = _pick_best_blend_alpha(
            proba_chain=proba_valid,
            y_true=fold_data["y_valid"],
            y_persist=fold_data["persist_valid_pred"],
            blend_alphas=blend_alphas,
            persist_metrics={
                "acc": fold_data["persist_valid_acc"],
                "macro_f1": fold_data["persist_valid_macro_f1"],
                "weighted_f1": fold_data["persist_valid_weighted_f1"],
                "macro_recall": fold_data["persist_valid_macro_recall"],
                "high_vol_recall": fold_data["persist_valid_high_vol_recall"],
            },
        )
    else:
        selected_alpha = 1.0
        pred_chain = np.argmax(proba_valid, axis=1).astype(int)
        m_chain = compute_metrics(fold_data["y_valid"], pred_chain)
        high_vol_recall = _high_vol_recall(m_chain)
        best_m = {
            "acc": float(m_chain.accuracy),
            "macro_f1": float(m_chain.macro_f1),
            "weighted_f1": float(m_chain.weighted_f1),
            "macro_recall": float(m_chain.macro_recall),
            "high_vol_recall": float(high_vol_recall),
            "delta_acc_vs_persistence": float(m_chain.accuracy - fold_data["persist_valid_acc"]),
            "delta_macro_f1_vs_persistence": float(
                m_chain.macro_f1 - fold_data["persist_valid_macro_f1"]
            ),
            "delta_weighted_f1_vs_persistence": float(
                m_chain.weighted_f1 - fold_data["persist_valid_weighted_f1"]
            ),
            "delta_macro_recall_vs_persistence": float(
                m_chain.macro_recall - fold_data["persist_valid_macro_recall"]
            ),
            "delta_high_vol_recall_vs_persistence": float(
                high_vol_recall - fold_data["persist_valid_high_vol_recall"]
            ),
        }

    return {
        "selected_alpha": float(selected_alpha),
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
        "n_features": int(fold_data["n_features"]),
        "n_valid": int(len(fold_data["y_valid"])),
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
    xgb_cfg: dict[str, Any],
    xgb_n_jobs: int,
    use_blend: bool,
    blend_alpha: float,
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
        garch_q=int(struct_cfg["garch_q"]),
        garch_dist=str(struct_cfg["garch_dist"]),
        garch_chain_agg=str(struct_cfg["garch_chain_agg"]),
        garch_scale=float(struct_cfg["garch_scale"]),
    )
    pack = make_dataset(dcfg)

    x_train = pack["train"][pack["feature_cols"]].to_numpy(dtype=np.float32)
    y_train = pack["train"]["regime"].to_numpy(dtype=int)
    x_valid = pack["valid"][pack["feature_cols"]].to_numpy(dtype=np.float32)
    y_valid = pack["valid"]["regime"].to_numpy(dtype=int)
    x_test = pack["test"][pack["feature_cols"]].to_numpy(dtype=np.float32)
    y_test = pack["test"]["regime"].to_numpy(dtype=int)

    cfg = XGBConfig(
        n_estimators=int(xgb_cfg["n_estimators"]),
        max_depth=int(xgb_cfg["max_depth"]),
        learning_rate=float(xgb_cfg["learning_rate"]),
        subsample=float(xgb_cfg["subsample"]),
        colsample_bytree=float(xgb_cfg["colsample_bytree"]),
        min_child_weight=float(xgb_cfg["min_child_weight"]),
        reg_lambda=float(xgb_cfg["reg_lambda"]),
        n_jobs=int(xgb_n_jobs),
        random_state=42,
        seed=42,
    )
    model = make_xgb_model(cfg)
    fit_xgb(model, x_train, y_train, X_valid=x_valid, y_valid=y_valid)

    aligned_test = _aligned_split_with_persistence(pack["df"], pack["test"], horizon=int(horizon))
    keep_test = np.asarray(aligned_test["keep_mask"], dtype=bool)
    x_test_eval = x_test[keep_test]
    y_test_eval = y_test[keep_test]
    y_persist_test = np.asarray(aligned_test["y_persist"], dtype=int)
    p_metrics = aligned_test["metrics"]

    proba_test = predict_proba_xgb(model, x_test_eval)
    if bool(use_blend):
        alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        persist_proba = _one_hot_probs(y_persist_test, n_classes=proba_test.shape[1])
        pred_test = np.argmax(alpha * proba_test + (1.0 - alpha) * persist_proba, axis=1).astype(int)
    else:
        alpha = 1.0
        pred_test = np.argmax(proba_test, axis=1).astype(int)
    m_chain = compute_metrics(y_test_eval, pred_test)
    chain_high_vol_recall = _high_vol_recall(m_chain)
    return {
        "selected_blend_alpha": alpha,
        "chain_test_acc": float(m_chain.accuracy),
        "chain_test_macro_f1": float(m_chain.macro_f1),
        "chain_test_weighted_f1": float(m_chain.weighted_f1),
        "chain_test_macro_recall": float(m_chain.macro_recall),
        "chain_test_high_vol_recall": float(chain_high_vol_recall),
        "persistence_test_acc": float(p_metrics["acc"]),
        "persistence_test_macro_f1": float(p_metrics["macro_f1"]),
        "persistence_test_weighted_f1": float(p_metrics["weighted_f1"]),
        "persistence_test_macro_recall": float(p_metrics["macro_recall"]),
        "persistence_test_high_vol_recall": float(p_metrics["high_vol_recall"]),
        "delta_test_acc_vs_persistence": float(m_chain.accuracy - p_metrics["acc"]),
        "delta_test_macro_f1_vs_persistence": float(m_chain.macro_f1 - p_metrics["macro_f1"]),
        "delta_test_weighted_f1_vs_persistence": float(
            m_chain.weighted_f1 - p_metrics["weighted_f1"]
        ),
        "delta_test_macro_recall_vs_persistence": float(
            m_chain.macro_recall - p_metrics["macro_recall"]
        ),
        "delta_test_high_vol_recall_vs_persistence": float(
            chain_high_vol_recall - p_metrics["high_vol_recall"]
        ),
        "n_test_eval": int(len(y_test_eval)),
        "n_features": int(len(pack["feature_cols"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward model selection for chain SARIMAX->GARCH->XGB."
    )
    parser.add_argument("--config_features", default="config/features.yaml")
    parser.add_argument("--config_sources", default="config/datasources.yaml")
    parser.add_argument("--horizon", type=int, default=20, choices=[5, 20])
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
    parser.add_argument("--use_blend", action="store_true")
    parser.add_argument(
        "--blend_alphas",
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated blend weights alpha for chain probs in alpha*chain + (1-alpha)*persistence.",
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

    features_cfg = load_yaml(args.config_features)
    sources_cfg = load_yaml(args.config_sources)
    db_path = sources_cfg["db"]["path"]
    regime_bins = int(features_cfg["targets"]["regime_bins"])
    tickers = tuple(args.tickers)
    if args.asset:
        tickers = (str(args.asset),)

    structural_cfgs = make_structural_configs(args.grid_profile)
    xgb_cfgs = make_xgb_configs(args.grid_profile)

    if int(args.max_struct_configs) > 0:
        structural_cfgs = structural_cfgs[: int(args.max_struct_configs)]
    if int(args.max_xgb_configs) > 0:
        xgb_cfgs = xgb_cfgs[: int(args.max_xgb_configs)]
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
                        res = _fit_eval_xgb(
                            xgb_cfg=xgb_cfgs[x_idx],
                            fold_data=fold_data,
                            xgb_n_jobs=int(args.xgb_n_jobs),
                            use_blend=bool(args.use_blend),
                            blend_alphas=blend_alphas,
                        )
                        combo_metrics[x_idx].append(res)
                        fold_rows.append(
                            {
                                "asset": asset_name,
                                "tickers": "|".join(group_tickers),
                                "struct_id": s_idx,
                                "xgb_id": x_idx,
                                "fold_id": fold_idx,
                                "train_end": train_end,
                                "valid_end": valid_end,
                                **s_cfg,
                                **xgb_cfgs[x_idx],
                                **res,
                            }
                        )
                    except Exception as e:
                        fold_rows.append(
                            {
                                "asset": asset_name,
                                "tickers": "|".join(group_tickers),
                                "struct_id": s_idx,
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
                macro_series = np.array([r["valid_macro_f1"] for r in rows], dtype=float)
                weighted_series = np.array([r["valid_weighted_f1"] for r in rows], dtype=float)
                macro_recall_series = np.array([r["valid_macro_recall"] for r in rows], dtype=float)
                high_vol_recall_series = np.array([r["valid_high_vol_recall"] for r in rows], dtype=float)
                delta_series = np.array([r["delta_macro_f1_vs_persistence"] for r in rows], dtype=float)
                delta_weighted_series = np.array(
                    [r["delta_weighted_f1_vs_persistence"] for r in rows], dtype=float
                )
                delta_macro_recall_series = np.array(
                    [r["delta_macro_recall_vs_persistence"] for r in rows], dtype=float
                )
                delta_high_vol_recall_series = np.array(
                    [r["delta_high_vol_recall_vs_persistence"] for r in rows], dtype=float
                )
                acc_series = np.array([r["valid_acc"] for r in rows], dtype=float)
                alpha_series = np.array([r["selected_alpha"] for r in rows], dtype=float)

                positive_rate = float(np.mean(delta_series > 0.0))
                stability_penalty = float(args.stability_lambda) * float(np.std(delta_series))
                positive_gap = max(0.0, float(args.min_positive_rate) - positive_rate)
                positive_penalty = float(args.positive_rate_lambda) * positive_gap
                robust_score = float(np.mean(delta_series)) - stability_penalty - positive_penalty

                summary_rows.append(
                    {
                        "asset": asset_name,
                        "tickers": "|".join(group_tickers),
                        "struct_id": s_idx,
                        "xgb_id": x_idx,
                        **s_cfg,
                        **xgb_cfgs[x_idx],
                        "n_folds_ok": int(len(rows)),
                        "mean_selected_alpha": float(np.mean(alpha_series)),
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
                "garch_q": best["garch_q"],
                "garch_dist": best["garch_dist"],
                "garch_chain_agg": best["garch_chain_agg"],
                "garch_scale": best["garch_scale"],
            },
            xgb_cfg={
                "n_estimators": best["n_estimators"],
                "max_depth": best["max_depth"],
                "learning_rate": best["learning_rate"],
                "subsample": best["subsample"],
                "colsample_bytree": best["colsample_bytree"],
                "min_child_weight": best["min_child_weight"],
                "reg_lambda": best["reg_lambda"],
            },
            xgb_n_jobs=int(args.xgb_n_jobs),
            use_blend=bool(args.use_blend),
            blend_alpha=float(best.get("mean_selected_alpha", 1.0)),
        )
        final_compare_rows.append(
            {
                "asset": asset_name,
                "tickers": "|".join(group_tickers),
                "horizon": int(args.horizon),
                "selected_struct_id": int(best["struct_id"]),
                "selected_xgb_id": int(best["xgb_id"]),
                **final_cmp,
            }
        )
        pd.DataFrame([final_compare_rows[-1]]).to_csv(
            outdir / "final_vs_persistence.csv", index=False
        )

        print(
            f"Asset={asset_name} | Folds={len(folds)} | Struct={len(structural_cfgs)} | XGB={len(xgb_cfgs)}"
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
