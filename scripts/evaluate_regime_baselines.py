'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-28

@description: Compare regime baselines and econometric+RF models with a common evaluation pipeline.
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quant_risk.datasets.make_dataset import DatasetConfig, build_xy, make_dataset
from quant_risk.models.baseline import (
    majority_class,
    persistence_pred_from_regime,
    predict_majority,
)
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


def _base_cfg(
    db_path: str,
    tickers: tuple[str, ...],
    horizon: int,
    train_end: str,
    valid_end: str,
    regime_bins: int,
    pooled: bool,
) -> dict[str, Any]:
    return dict(
        db_path=db_path,
        tickers=tickers,
        horizon=int(horizon),
        pooled=bool(pooled),
        train_end=train_end,
        valid_end=valid_end,
        regime_bins=int(regime_bins),
    )


def _metrics_record(
    model_name: str,
    horizon: int,
    split_name: str,
    y_true,
    y_pred,
) -> dict[str, Any]:
    m = compute_metrics(y_true, y_pred)
    return {
        "model": model_name,
        "horizon": int(horizon),
        "split": split_name,
        "accuracy": float(m.accuracy),
        "macro_f1": float(m.macro_f1),
        "weighted_f1": float(m.weighted_f1),
        "n_eval": int(len(y_true)),
        "confusion_matrix": m.confusion.tolist(),
    }


def _eval_majority(pack: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    train = pack["train"]
    valid = pack["valid"]
    test = pack["test"]

    maj = majority_class(train["regime"].to_numpy())
    yhat_valid = predict_majority(len(valid), maj)
    yhat_test = predict_majority(len(test), maj)

    return [
        _metrics_record("MajorityClass", horizon, "valid", valid["regime"], yhat_valid),
        _metrics_record("MajorityClass", horizon, "test", test["regime"], yhat_test),
    ]


def _eval_persistence(pack: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    df_full = pack["df"][["ticker", "date", "regime"]].copy()
    pred = persistence_pred_from_regime(df_full, horizon=horizon, regime_col="regime")
    df_full["persist_pred"] = pred

    out = []
    for split_name, split_df in [("valid", pack["valid"]), ("test", pack["test"])]:
        keys = split_df[["ticker", "date", "regime"]].copy()
        merged = keys.merge(
            df_full[["ticker", "date", "persist_pred"]],
            on=["ticker", "date"],
            how="left",
        )
        merged = merged.dropna(subset=["persist_pred"]).copy()
        y_true = merged["regime"].to_numpy(dtype=int)
        y_pred = merged["persist_pred"].to_numpy(dtype=int)
        out.append(_metrics_record("Persistence_t_minus_h", horizon, split_name, y_true, y_pred))
    return out


def _eval_rf_model(model_name: str, pack: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    from sklearn.ensemble import RandomForestClassifier

    feature_cols = pack["feature_cols"]
    x_train, y_train = build_xy(pack["train"], feature_cols)
    x_valid, y_valid = build_xy(pack["valid"], feature_cols)
    x_test, y_test = build_xy(pack["test"], feature_cols)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)

    pred_valid = model.predict(x_valid)
    pred_test = model.predict(x_test)
    return [
        _metrics_record(model_name, horizon, "valid", y_valid, pred_valid),
        _metrics_record(model_name, horizon, "test", y_test, pred_test),
    ]


def _eval_current_chain_xgb(
    base_cfg: dict[str, Any],
    horizon: int,
) -> list[dict[str, Any]]:
    # Fixed "current model" configs from recent chain experiments.
    if int(horizon) == 5:
        chain_cfg = dict(
            sarimax_order=(1, 0, 1),
            sarimax_chain_exog_cols=(),
            garch_p=1,
            garch_q=1,
            garch_chain_agg="rms",
            garch_scale=100.0,
        )
    else:
        chain_cfg = dict(
            sarimax_order=(2, 0, 1),
            sarimax_chain_exog_cols=("net_liquidity_diff",),
            garch_p=2,
            garch_q=2,
            garch_chain_agg="rms",
            garch_scale=100.0,
        )

    cfg = DatasetConfig(
        **base_cfg,
        use_sarimax_garch_chain=True,
        garch_dist="tstudent",
        **chain_cfg,
    )
    pack = make_dataset(cfg)
    feature_cols = pack["feature_cols"]
    x_train, y_train = build_xy(pack["train"], feature_cols)
    x_valid, y_valid = build_xy(pack["valid"], feature_cols)
    x_test, y_test = build_xy(pack["test"], feature_cols)

    model_cfg = XGBConfig(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1.0,
        reg_lambda=1.0,
        random_state=42,
        seed=42,
    )
    model = make_xgb_model(model_cfg)
    fit_xgb(model, x_train, y_train, X_valid=x_valid, y_valid=y_valid)

    pred_valid = predict_xgb(model, x_valid)
    pred_test = predict_xgb(model, x_test)
    return [
        _metrics_record("Current_Chain_XGB_tstudent", horizon, "valid", y_valid, pred_valid),
        _metrics_record("Current_Chain_XGB_tstudent", horizon, "test", y_test, pred_test),
    ]


def _split_class_balance(pack: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ["train", "valid", "test"]:
        split_df = pack[split_name]
        vc = split_df["regime"].value_counts(normalize=True).sort_index()
        rows.append(
            {
                "horizon": int(horizon),
                "split": split_name,
                "class_0": float(vc.get(0, 0.0)),
                "class_1": float(vc.get(1, 0.0)),
                "class_2": float(vc.get(2, 0.0)),
                "majority_share": float(vc.max() if len(vc) else np.nan),
            }
        )
    return rows


def build_improvement_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    out_rows = []

    for horizon in sorted(test_df["horizon"].unique()):
        sub = test_df[test_df["horizon"] == horizon].copy()
        maj = sub[sub["model"] == "MajorityClass"].iloc[0]
        per = sub[sub["model"] == "Persistence_t_minus_h"].iloc[0]
        for _, row in sub.iterrows():
            if row["model"] in {"MajorityClass", "Persistence_t_minus_h"}:
                continue
            out_rows.append(
                {
                    "model": row["model"],
                    "horizon": int(horizon),
                    "delta_acc_vs_majority": float(row["accuracy"] - maj["accuracy"]),
                    "delta_macro_f1_vs_majority": float(row["macro_f1"] - maj["macro_f1"]),
                    "delta_weighted_f1_vs_majority": float(row["weighted_f1"] - maj["weighted_f1"]),
                    "delta_acc_vs_persistence": float(row["accuracy"] - per["accuracy"]),
                    "delta_macro_f1_vs_persistence": float(row["macro_f1"] - per["macro_f1"]),
                    "delta_weighted_f1_vs_persistence": float(row["weighted_f1"] - per["weighted_f1"]),
                }
            )
    return pd.DataFrame(out_rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate regime baselines and econometric+RF models with a shared pipeline."
    )
    parser.add_argument("--config_features", default="config/features.yaml")
    parser.add_argument("--config_sources", default="config/datasources.yaml")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20])
    parser.add_argument("--tickers", nargs="+", default=["^GSPC", "BTC-USD", "TLT"])
    parser.add_argument("--pooled", action="store_true")
    parser.add_argument("--outdir", default="runs/baseline_eval")
    args = parser.parse_args()

    features_cfg = load_yaml(args.config_features)
    sources_cfg = load_yaml(args.config_sources)
    db_path = sources_cfg["db"]["path"]
    train_end = features_cfg["split"]["train_end"]
    valid_end = features_cfg["split"]["val_end"]
    regime_bins = int(features_cfg["targets"]["regime_bins"])

    all_metrics: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []

    for horizon in args.horizons:
        print(f"\n[run] horizon={horizon}")
        base_cfg = _base_cfg(
            db_path=db_path,
            tickers=tuple(args.tickers),
            horizon=int(horizon),
            train_end=train_end,
            valid_end=valid_end,
            regime_bins=regime_bins,
            pooled=bool(args.pooled),
        )

        base_pack = make_dataset(DatasetConfig(**base_cfg))
        balance_rows.extend(_split_class_balance(base_pack, horizon))
        all_metrics.extend(_eval_majority(base_pack, horizon))
        all_metrics.extend(_eval_persistence(base_pack, horizon))

        sarimax_pack = make_dataset(DatasetConfig(**base_cfg, use_sarimax=True))
        all_metrics.extend(_eval_rf_model("SARIMAX_plus_RF", sarimax_pack, horizon))

        garch_pack = make_dataset(
            DatasetConfig(
                **base_cfg,
                use_garch=True,
                garch_agg="rms",
            )
        )
        all_metrics.extend(_eval_rf_model("GARCH_plus_RF", garch_pack, horizon))

        all_metrics.extend(_eval_current_chain_xgb(base_cfg, horizon))

    metrics_df = pd.DataFrame(all_metrics).sort_values(["horizon", "split", "model"])
    test_table = metrics_df[metrics_df["split"] == "test"][
        ["model", "horizon", "accuracy", "macro_f1", "weighted_f1", "n_eval"]
    ].sort_values(["horizon", "macro_f1"], ascending=[True, False])
    improvement_df = build_improvement_table(metrics_df)
    balance_df = pd.DataFrame(balance_rows).sort_values(["horizon", "split"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(outdir / "metrics_all_splits.csv", index=False)
    test_table.to_csv(outdir / "comparison_test.csv", index=False)
    improvement_df.to_csv(outdir / "improvements_vs_baselines.csv", index=False)
    balance_df.to_csv(outdir / "class_balance.csv", index=False)

    payload = {
        "config": {
            "db_path": db_path,
            "horizons": [int(h) for h in args.horizons],
            "tickers": list(args.tickers),
            "pooled": bool(args.pooled),
            "train_end": train_end,
            "valid_end": valid_end,
            "regime_bins": regime_bins,
        },
        "files": {
            "metrics_all_splits": str(outdir / "metrics_all_splits.csv"),
            "comparison_test": str(outdir / "comparison_test.csv"),
            "improvements_vs_baselines": str(outdir / "improvements_vs_baselines.csv"),
            "class_balance": str(outdir / "class_balance.csv"),
        },
    }
    with open(outdir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== Test Comparison ===")
    print(test_table.to_string(index=False))
    print("\n=== Improvements vs Baselines ===")
    print(improvement_df.to_string(index=False))
    print("\nSaved outputs to:", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
