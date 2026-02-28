'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-28

@description: Walk-forward model selection for chained SARIMAX->GARCH->XGB
using mean/stability of delta macro F1 vs persistence baseline.
'''

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_risk.datasets.make_dataset import DatasetConfig, build_xy, load_joined, make_dataset
from quant_risk.models.baseline import persistence_pred_from_regime
from quant_risk.models.metrics import compute_metrics
from quant_risk.models.tabular.xgb import (
    XGBConfig,
    fit as fit_xgb,
    make_model as make_xgb_model,
    predict as predict_xgb,
)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp("M")


def build_folds(
    dates: pd.Series,
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


def persistence_metrics_from_pack(pack: dict[str, Any], horizon: int) -> tuple[float, float]:
    df_full = pack["df"][["ticker", "date", "regime"]].copy()
    df_full["persist_pred"] = persistence_pred_from_regime(df_full, horizon=horizon, regime_col="regime")

    valid = pack["valid"][["ticker", "date", "regime"]].copy()
    merged = valid.merge(
        df_full[["ticker", "date", "persist_pred"]],
        on=["ticker", "date"],
        how="left",
    ).dropna(subset=["persist_pred"])

    y_true = merged["regime"].to_numpy(dtype=int)
    y_pred = merged["persist_pred"].to_numpy(dtype=int)
    m = compute_metrics(y_true, y_pred)
    return float(m.accuracy), float(m.macro_f1)


def evaluate_experiment_on_fold(
    db_path: str,
    tickers: tuple[str, ...],
    horizon: int,
    regime_bins: int,
    fold_train_end: str,
    fold_valid_end: str,
    exp: dict[str, Any],
) -> dict[str, Any]:
    dcfg = DatasetConfig(
        db_path=db_path,
        tickers=tickers,
        horizon=int(horizon),
        pooled=False,
        train_end=fold_train_end,
        valid_end=fold_valid_end,
        regime_bins=int(regime_bins),
        use_sarimax_garch_chain=True,
        sarimax_order=tuple(exp["sarimax_order"]),
        sarimax_chain_exog_cols=tuple(exp["sarimax_chain_exog_cols"]),
        garch_p=int(exp["garch_p"]),
        garch_q=int(exp["garch_q"]),
        garch_dist=str(exp["garch_dist"]),
        garch_chain_agg=str(exp["garch_chain_agg"]),
        garch_scale=float(exp["garch_scale"]),
    )

    pack = make_dataset(dcfg)
    X_train, y_train = build_xy(pack["train"], pack["feature_cols"])
    X_valid, y_valid = build_xy(pack["valid"], pack["feature_cols"])

    model_cfg = XGBConfig(
        n_estimators=int(exp["n_estimators"]),
        max_depth=int(exp["max_depth"]),
        learning_rate=float(exp["learning_rate"]),
        subsample=float(exp["subsample"]),
        colsample_bytree=float(exp["colsample_bytree"]),
        min_child_weight=float(exp["min_child_weight"]),
        reg_lambda=float(exp["reg_lambda"]),
        random_state=42,
        seed=42,
    )
    model = make_xgb_model(model_cfg)
    fit_xgb(model, X_train, y_train, X_valid=X_valid, y_valid=y_valid)
    y_pred = predict_xgb(model, X_valid)
    m_model = compute_metrics(y_valid, y_pred)

    p_acc, p_macro = persistence_metrics_from_pack(pack, horizon=horizon)
    return {
        "valid_acc": float(m_model.accuracy),
        "valid_macro_f1": float(m_model.macro_f1),
        "valid_weighted_f1": float(m_model.weighted_f1),
        "delta_acc_vs_persistence": float(m_model.accuracy - p_acc),
        "delta_macro_f1_vs_persistence": float(m_model.macro_f1 - p_macro),
        "n_features": int(len(pack["feature_cols"])),
        "n_valid": int(len(y_valid)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward model selection for chain SARIMAX->GARCH->XGB."
    )
    parser.add_argument("--config_features", default="config/features.yaml")
    parser.add_argument("--config_sources", default="config/datasources.yaml")
    parser.add_argument("--horizon", type=int, default=20, choices=[5, 20])
    parser.add_argument("--tickers", nargs="+", default=["^GSPC", "BTC-USD", "TLT"])
    parser.add_argument("--min_train_end", default="2018-12-31")
    parser.add_argument("--max_valid_end", default="2023-12-31")
    parser.add_argument("--valid_months", type=int, default=12)
    parser.add_argument("--step_months", type=int, default=6)
    parser.add_argument("--outdir", default="runs/walk_forward_chain_xgb")
    parser.add_argument("--max_experiments", type=int, default=0)
    args = parser.parse_args()

    features_cfg = load_yaml(args.config_features)
    sources_cfg = load_yaml(args.config_sources)
    db_path = sources_cfg["db"]["path"]
    regime_bins = int(features_cfg["targets"]["regime_bins"])
    tickers = tuple(args.tickers)

    joined = load_joined(
        DatasetConfig(
            db_path=db_path,
            tickers=tickers,
            horizon=int(args.horizon),
            pooled=False,
        )
    )
    folds = build_folds(
        dates=joined["date"],
        min_train_end=args.min_train_end,
        max_valid_end=args.max_valid_end,
        valid_months=int(args.valid_months),
        step_months=int(args.step_months),
    )
    if not folds:
        raise SystemExit("No folds generated. Check date range arguments.")

    grid = {
        "sarimax_order": [(1, 0, 1), (2, 0, 1)],
        "sarimax_chain_exog_cols": [(), ("net_liquidity_diff",)],
        "garch_p": [1, 2],
        "garch_q": [1, 2],
        "garch_dist": ["tstudent"],
        "garch_chain_agg": ["rms", "mean"],
        "garch_scale": [80.0, 100.0],
        "n_estimators": [300, 500],
        "max_depth": [4, 5],
        "learning_rate": [0.05],
        "subsample": [0.9],
        "colsample_bytree": [0.9],
        "min_child_weight": [1.0],
        "reg_lambda": [1.0],
    }
    keys = list(grid.keys())
    experiments = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    if int(args.max_experiments) > 0:
        experiments = experiments[: int(args.max_experiments)]

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for exp_idx, exp in enumerate(experiments):
        fold_metrics = []
        for fold_idx, (train_end, valid_end) in enumerate(folds):
            try:
                res = evaluate_experiment_on_fold(
                    db_path=db_path,
                    tickers=tickers,
                    horizon=int(args.horizon),
                    regime_bins=regime_bins,
                    fold_train_end=train_end,
                    fold_valid_end=valid_end,
                    exp=exp,
                )
                fold_metrics.append(res)
                fold_rows.append(
                    {
                        "experiment_id": exp_idx,
                        "fold_id": fold_idx,
                        "train_end": train_end,
                        "valid_end": valid_end,
                        **exp,
                        **res,
                    }
                )
            except Exception as e:
                fold_rows.append(
                    {
                        "experiment_id": exp_idx,
                        "fold_id": fold_idx,
                        "train_end": train_end,
                        "valid_end": valid_end,
                        **exp,
                        "error": repr(e),
                    }
                )

        if not fold_metrics:
            continue

        macro_series = np.array([m["valid_macro_f1"] for m in fold_metrics], dtype=float)
        delta_series = np.array([m["delta_macro_f1_vs_persistence"] for m in fold_metrics], dtype=float)
        acc_series = np.array([m["valid_acc"] for m in fold_metrics], dtype=float)
        summary_rows.append(
            {
                "experiment_id": exp_idx,
                **exp,
                "n_folds_ok": int(len(fold_metrics)),
                "mean_valid_acc": float(np.mean(acc_series)),
                "mean_valid_macro_f1": float(np.mean(macro_series)),
                "std_valid_macro_f1": float(np.std(macro_series)),
                "mean_delta_macro_vs_persistence": float(np.mean(delta_series)),
                "std_delta_macro_vs_persistence": float(np.std(delta_series)),
                "stability_score": float(np.mean(delta_series) - np.std(delta_series)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise SystemExit("No successful experiments in walk-forward.")

    summary_df = summary_df.sort_values(
        ["mean_delta_macro_vs_persistence", "std_delta_macro_vs_persistence", "mean_valid_macro_f1"],
        ascending=[False, True, False],
    )
    best = summary_df.iloc[0].to_dict()

    outdir = Path(args.outdir) / f"h{args.horizon}"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(outdir / "fold_metrics.csv", index=False)
    summary_df.to_csv(outdir / "summary.csv", index=False)
    with open(outdir / "best.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    print(f"Folds: {len(folds)} | Experiments: {len(experiments)}")
    print(summary_df.head(10).to_string(index=False))
    print("\nSaved:", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
