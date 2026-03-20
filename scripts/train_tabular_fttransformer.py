'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-27

@description: Train FT-Transformer tabular model for volatility-regime classification.
'''

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from quant_risk.config import load_yaml, resolve_tickers
from quant_risk.datasets.make_dataset import DatasetConfig, build_xy, make_dataset
from quant_risk.models.metrics import compute_metrics, per_class_accuracy
from quant_risk.models.tabular.common import build_variant_name, set_global_seed
from quant_risk.models.tabular.ft_transformer import (
    FTTransformerConfig,
    fit,
    make_model,
    predict,
)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def bins_to_jsonable(bins):
    if hasattr(bins, "to_dict"):
        return bins.to_dict(orient="index")
    return bins


def train_eval_one(model, x_train, y_train, x_valid, y_valid, x_test, y_test) -> dict:
    fit(model, x_train, y_train, X_valid=x_valid, y_valid=y_valid)

    pred_valid = predict(model, x_valid)
    pred_test = predict(model, x_test)

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
    parser = argparse.ArgumentParser(description="Train FT-Transformer tabular regime classifier.")
    parser.add_argument("--config_features", default="config/features.yaml")
    parser.add_argument("--config_sources", default="config/datasources.yaml")
    parser.add_argument("--horizon", type=int, default=20, choices=[5, 20])
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--pooled", action="store_true")
    parser.add_argument("--use_sarimax", action="store_true")
    parser.add_argument("--use_garch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--d_token", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--ffn_multiplier", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--outdir", default="runs/tabular/fttransformer")
    args = parser.parse_args()

    features_cfg = load_yaml(args.config_features)
    sources_cfg = load_yaml(args.config_sources)
    db_path = sources_cfg["db"]["path"]
    tickers = resolve_tickers(args.tickers, args.config_sources)
    train_end = features_cfg["split"]["train_end"]
    valid_end = features_cfg["split"]["val_end"]
    regime_bins = int(features_cfg["targets"]["regime_bins"])

    set_global_seed(args.seed, use_torch=True)

    variant_name = build_variant_name(
        pooled=bool(args.pooled),
        use_sarimax=bool(args.use_sarimax),
        use_garch=bool(args.use_garch),
    )
    variant_dir = Path(args.outdir) / f"h{args.horizon}" / f"variant_{variant_name}"
    ensure_dir(variant_dir)

    dataset_cfg = DatasetConfig(
        db_path=db_path,
        tickers=tickers,
        horizon=args.horizon,
        pooled=bool(args.pooled),
        train_end=train_end,
        valid_end=valid_end,
        regime_bins=regime_bins,
        use_sarimax=bool(args.use_sarimax),
        use_garch=bool(args.use_garch),
    )
    pack = make_dataset(dataset_cfg)
    feature_cols = pack["feature_cols"]
    bins = bins_to_jsonable(pack.get("bins"))

    x_train, y_train = build_xy(pack["train"], feature_cols)
    x_valid, y_valid = build_xy(pack["valid"], feature_cols)
    x_test, y_test = build_xy(pack["test"], feature_cols)

    model_cfg = FTTransformerConfig(
        d_token=args.d_token,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_multiplier=args.ffn_multiplier,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )

    try:
        model = make_model(model_cfg)
    except ImportError as e:
        print(str(e))
        return 1

    metrics = train_eval_one(model, x_train, y_train, x_valid, y_valid, x_test, y_test)
    save_json(variant_dir / "metrics.json", metrics)

    print(
        f"VALID acc={metrics['valid']['accuracy']:.4f} macroF1={metrics['valid']['macro_f1']:.4f}"
    )
    print(metrics["valid"]["report"])
    print(
        f"TEST  acc={metrics['test']['accuracy']:.4f} macroF1={metrics['test']['macro_f1']:.4f}"
    )
    print(metrics["test"]["report"])

    config_json = {
        "variant": variant_name,
        "dataset": {
            "db_path": db_path,
            "tickers": list(tickers),
            "horizon": int(args.horizon),
            "pooled": bool(args.pooled),
            "train_end": train_end,
            "valid_end": valid_end,
            "regime_bins": regime_bins,
            "use_sarimax": bool(args.use_sarimax),
            "use_garch": bool(args.use_garch),
            "feature_cols": feature_cols,
            "bins": bins,
        },
        "model": {
            "name": "ft_transformer",
            "params": asdict(model_cfg),
        },
        "seed": int(args.seed),
    }
    save_json(variant_dir / "config.json", config_json)

    summary_row = {
        "variant": variant_name,
        "model": "ft_transformer",
        "valid_acc": metrics["valid"]["accuracy"],
        "valid_macro_f1": metrics["valid"]["macro_f1"],
        "test_acc": metrics["test"]["accuracy"],
        "test_macro_f1": metrics["test"]["macro_f1"],
        "n_features": len(feature_cols),
        "pooled": bool(args.pooled),
        "use_sarimax": bool(args.use_sarimax),
        "use_garch": bool(args.use_garch),
        "seed": int(args.seed),
    }
    pd.DataFrame([summary_row]).to_csv(
        Path(args.outdir) / f"h{args.horizon}" / "summary.csv",
        index=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
