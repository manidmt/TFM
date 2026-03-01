'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-27

@description: Train chained SARIMAX->GARCH->Tabular models for regime classification.
'''

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import yaml

from quant_risk.datasets.make_dataset import DatasetConfig, build_xy, make_dataset
from quant_risk.models.metrics import compute_metrics, per_class_accuracy
from quant_risk.models.tabular.common import set_global_seed
from quant_risk.models.tabular.xgb import (
    XGBConfig,
    fit as fit_xgb,
    make_model as make_xgb_model,
    predict as predict_xgb,
)
from quant_risk.models.tabular.tabnet import (
    TabNetConfig,
    fit as fit_tabnet,
    make_model as make_tabnet_model,
    predict as predict_tabnet,
)
from quant_risk.models.tabular.ft_transformer import (
    FTTransformerConfig,
    fit as fit_ftt,
    make_model as make_ftt_model,
    predict as predict_ftt,
)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def bins_to_jsonable(bins):
    if hasattr(bins, "to_dict"):
        return bins.to_dict(orient="index")
    return bins


def parse_chain_exog(value: str) -> tuple[str, ...]:
    v = str(value).strip().lower()
    if v == "none":
        return ()
    if v == "nl":
        return ("net_liquidity_diff",)
    raise SystemExit("Invalid --chain_exog. Use: none | nl")


def build_variant_name(pooled: bool, chain_exog: str, p: int, q: int, agg: str) -> str:
    scope = "pooled" if pooled else "single"
    return f"{scope}_chain_{chain_exog}_p{p}q{q}_agg-{agg}"


def build_model(args):
    if args.model == "xgb":
        cfg = XGBConfig(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            min_child_weight=args.min_child_weight,
            reg_lambda=args.reg_lambda,
            random_state=args.seed,
            seed=args.seed,
        )
        return cfg, make_xgb_model(cfg), fit_xgb, predict_xgb

    if args.model == "tabnet":
        cfg = TabNetConfig(
            n_d=args.n_d,
            n_a=args.n_a,
            n_steps=args.n_steps,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            virtual_batch_size=args.virtual_batch_size,
            learning_rate=args.learning_rate_tabnet,
            weight_decay=args.weight_decay_tabnet,
            seed=args.seed,
        )
        return cfg, make_tabnet_model(cfg), fit_tabnet, predict_tabnet

    if args.model == "ftt":
        cfg = FTTransformerConfig(
            d_token=args.d_token,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ffn_multiplier=args.ffn_multiplier,
            dropout=args.dropout,
            learning_rate=args.learning_rate_ftt,
            weight_decay=args.weight_decay_ftt,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
        )
        return cfg, make_ftt_model(cfg), fit_ftt, predict_ftt

    raise SystemExit("Invalid --model. Use: xgb | tabnet | ftt")


def train_eval_one(model, fit_fn, predict_fn, x_train, y_train, x_valid, y_valid, x_test, y_test) -> dict:
    fit_fn(model, x_train, y_train, X_valid=x_valid, y_valid=y_valid)

    pred_valid = predict_fn(model, x_valid)
    pred_test = predict_fn(model, x_test)

    metrics_valid = compute_metrics(y_valid, pred_valid)
    metrics_test = compute_metrics(y_test, pred_test)

    return {
        "valid": {
            "accuracy": metrics_valid.accuracy,
            "macro_f1": metrics_valid.macro_f1,
            "report": metrics_valid.report,
            "per_class_acc": per_class_accuracy(metrics_valid.confusion),
            "confusion": metrics_valid.confusion.tolist(),
        },
        "test": {
            "accuracy": metrics_test.accuracy,
            "macro_f1": metrics_test.macro_f1,
            "report": metrics_test.report,
            "per_class_acc": per_class_accuracy(metrics_test.confusion),
            "confusion": metrics_test.confusion.tolist(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train chained SARIMAX->GARCH->Tabular model for regime classification."
    )
    parser.add_argument("--model", required=True, choices=["xgb", "tabnet", "ftt"])
    parser.add_argument("--config_features", default="config/features.yaml")
    parser.add_argument("--config_sources", default="config/datasources.yaml")
    parser.add_argument("--horizon", type=int, default=20, choices=[5, 20])
    parser.add_argument("--tickers", nargs="+", default=["^GSPC", "BTC-USD", "TLT"])
    parser.add_argument("--pooled", action="store_true")
    parser.add_argument("--chain_exog", default="none", choices=["none", "nl"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sarimax_order", default="1,0,1")
    parser.add_argument("--sarimax_seasonal_order", default="0,0,0,0")
    parser.add_argument("--sarimax_trend", default="c")
    parser.add_argument("--garch_p", type=int, default=1)
    parser.add_argument("--garch_q", type=int, default=1)
    parser.add_argument("--garch_scale", type=float, default=100.0)
    parser.add_argument("--garch_agg", default="rms", choices=["last", "mean", "rms"])

    # xgb
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)

    # tabnet
    parser.add_argument("--n_d", type=int, default=16)
    parser.add_argument("--n_a", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=4)
    parser.add_argument("--virtual_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate_tabnet", type=float, default=0.02)
    parser.add_argument("--weight_decay_tabnet", type=float, default=1e-5)

    # ftt
    parser.add_argument("--d_token", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--ffn_multiplier", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate_ftt", type=float, default=1e-3)
    parser.add_argument("--weight_decay_ftt", type=float, default=1e-4)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--outdir", default="runs/hybrid_chain/sarimax_garch")
    args = parser.parse_args()

    features_cfg = load_yaml(args.config_features)
    sources_cfg = load_yaml(args.config_sources)
    db_path = sources_cfg["db"]["path"]
    train_end = features_cfg["split"]["train_end"]
    valid_end = features_cfg["split"]["val_end"]
    regime_bins = int(features_cfg["targets"]["regime_bins"])

    sarimax_order = tuple(int(x.strip()) for x in args.sarimax_order.split(","))
    sarimax_seasonal_order = tuple(int(x.strip()) for x in args.sarimax_seasonal_order.split(","))
    if len(sarimax_order) != 3 or len(sarimax_seasonal_order) != 4:
        raise SystemExit("Invalid SARIMAX order/seasonal_order.")

    set_global_seed(args.seed, use_torch=(args.model in {"tabnet", "ftt"}))
    exog_cols = parse_chain_exog(args.chain_exog)

    variant_name = build_variant_name(
        pooled=bool(args.pooled),
        chain_exog=args.chain_exog,
        p=int(args.garch_p),
        q=int(args.garch_q),
        agg=str(args.garch_agg),
    )
    output_root = Path(args.outdir) / args.model / f"h{args.horizon}"
    variant_dir = output_root / f"variant_{variant_name}"
    ensure_dir(variant_dir)

    dataset_cfg = DatasetConfig(
        db_path=db_path,
        tickers=tuple(args.tickers),
        horizon=int(args.horizon),
        pooled=bool(args.pooled),
        train_end=train_end,
        valid_end=valid_end,
        regime_bins=regime_bins,
        use_sarimax_garch_chain=True,
        sarimax_order=sarimax_order,
        sarimax_seasonal_order=sarimax_seasonal_order,
        sarimax_trend=args.sarimax_trend,
        sarimax_chain_exog_cols=exog_cols,
        garch_p=int(args.garch_p),
        garch_q=int(args.garch_q),
        garch_mean="zero",
        garch_target_col="sarimax_resid",
        garch_scale=float(args.garch_scale),
        garch_chain_agg=str(args.garch_agg),
    )

    pack = make_dataset(dataset_cfg)
    feature_cols = pack["feature_cols"]
    bins = pack.get("bins")
    if bins is not None and hasattr(bins, "to_csv"):
        bins.to_csv(variant_dir / "bins.csv")
    (variant_dir / "feature_cols.txt").write_text("\n".join(feature_cols), encoding="utf-8")

    x_train, y_train = build_xy(pack["train"], feature_cols)
    x_valid, y_valid = build_xy(pack["valid"], feature_cols)
    x_test, y_test = build_xy(pack["test"], feature_cols)

    try:
        model_cfg, model, fit_fn, predict_fn = build_model(args)
    except ImportError as e:
        print(str(e))
        return 1

    metrics = train_eval_one(
        model=model,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        y_test=y_test,
    )
    save_json(variant_dir / "metrics.json", metrics)

    print(f"VALID acc={metrics['valid']['accuracy']:.4f} macroF1={metrics['valid']['macro_f1']:.4f}")
    print(metrics["valid"]["report"])
    print(f"TEST  acc={metrics['test']['accuracy']:.4f} macroF1={metrics['test']['macro_f1']:.4f}")
    print(metrics["test"]["report"])

    config_json = {
        "variant": variant_name,
        "dataset": {
            "db_path": db_path,
            "tickers": list(args.tickers),
            "horizon": int(args.horizon),
            "pooled": bool(args.pooled),
            "train_end": train_end,
            "valid_end": valid_end,
            "regime_bins": regime_bins,
            "use_sarimax_garch_chain": True,
            "sarimax_chain_exog_cols": list(exog_cols),
            "sarimax_target_col": "logret",
            "sarimax_log_transform": False,
            "garch_target_col": "sarimax_resid",
            "garch_mean": "zero",
            "garch_agg": str(args.garch_agg),
            "feature_cols": feature_cols,
            "bins": bins_to_jsonable(bins),
        },
        "model": {
            "name": args.model,
            "params": asdict(model_cfg),
        },
        "seed": int(args.seed),
    }
    save_json(variant_dir / "config.json", config_json)

    summary_row = {
        "variant": variant_name,
        "model": args.model,
        "valid_acc": metrics["valid"]["accuracy"],
        "valid_macro_f1": metrics["valid"]["macro_f1"],
        "test_acc": metrics["test"]["accuracy"],
        "test_macro_f1": metrics["test"]["macro_f1"],
        "n_features": len(feature_cols),
        "pooled": bool(args.pooled),
        "chain_exog": args.chain_exog,
        "garch_p": int(args.garch_p),
        "garch_q": int(args.garch_q),
        "garch_agg": str(args.garch_agg),
        "seed": int(args.seed),
    }
    pd.DataFrame([summary_row]).to_csv(output_root / "summary.csv", index=False)
    print("\nSaved summary ->", output_root / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
