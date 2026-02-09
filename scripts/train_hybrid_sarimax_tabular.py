'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-10

@description: Train hybrid volatility-regime classifiers with SARIMAX + tabular features.
Runs multiple SARIMAX exog variants and persists metrics/artifacts.
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from quant_risk.datasets.make_dataset import DatasetConfig, build_xy, make_dataset
from quant_risk.models.metrics import compute_metrics, per_class_accuracy


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def train_eval_one(model, x_train, y_train, x_valid, y_valid, x_test, y_test) -> dict:
    model.fit(x_train, y_train)

    pred_valid = model.predict(x_valid)
    pred_test = model.predict(x_test)

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
    arg_parser = argparse.ArgumentParser(
        description="Train hybrid SARIMAX + tabular models for regime classification."
    )
    arg_parser.add_argument("--config_features", default="config/features.yaml")
    arg_parser.add_argument("--config_sources", default="config/datasources.yaml")
    arg_parser.add_argument("--horizon", type=int, default=20, choices=[5, 20])
    arg_parser.add_argument("--tickers", nargs="+", default=["^GSPC", "BTC-USD", "TLT"])
    arg_parser.add_argument("--pooled", action="store_true")

    # Variants:
    #   --sarimax_variants none,nl
    # where "none" uses exog_cols=()
    # and "nl" uses exog_cols=("net_liquidity_diff",)
    arg_parser.add_argument("--sarimax_variants", default="none,nl")
    arg_parser.add_argument("--net_liquidity_col", default="net_liquidity_diff")

    # model params
    arg_parser.add_argument("--rf_estimators", type=int, default=400)
    arg_parser.add_argument("--rf_min_leaf", type=int, default=5)
    arg_parser.add_argument("--outdir", default="runs/hybrid_sarimax")

    parsed_args = arg_parser.parse_args()

    features_config_dict = load_yaml(parsed_args.config_features)
    sources_config_dict = load_yaml(parsed_args.config_sources)
    database_path = sources_config_dict["db"]["path"]

    output_root = Path(parsed_args.outdir) / f"h{parsed_args.horizon}"
    ensure_dir(output_root)

    sarimax_variants = [v.strip() for v in parsed_args.sarimax_variants.split(",") if v.strip()]
    if not sarimax_variants:
        raise SystemExit("No variants provided.")

    # Build base dataset config fields
    base_kwargs = dict(
        db_path=database_path,
        tickers=tuple(parsed_args.tickers),
        horizon=parsed_args.horizon,
        pooled=bool(parsed_args.pooled),
        train_end=features_config_dict["split"]["train_end"],
        valid_end=features_config_dict["split"]["val_end"],
        regime_bins=features_config_dict["targets"]["regime_bins"],
        use_sarimax=True,
        sarimax_order=(1, 0, 1),
        sarimax_seasonal_order=(0, 0, 0, 0),
        sarimax_trend="c",
        sarimax_log_transform=True,
    )

    summary_records = []

    for variant in sarimax_variants:
        if variant == "none":
            sarimax_exog_cols = ()
        elif variant == "nl":
            sarimax_exog_cols = (parsed_args.net_liquidity_col,)
        else:
            # allow custom variant: v can be "exog:col1|col2"
            if variant.startswith("exog:"):
                cols = variant.split(":", 1)[1]
                sarimax_exog_cols = tuple([c for c in cols.split("|") if c])
            else:
                raise SystemExit(
                    f"Unknown variant '{variant}'. Use none,nl or exog:col1|col2"
                )

        variant_dir = output_root / f"variant_{variant}"
        ensure_dir(variant_dir)

        dataset_cfg = DatasetConfig(**base_kwargs, sarimax_exog_cols=tuple(sarimax_exog_cols))

        print("\n" + "=" * 80)
        print(f"Variant={variant} | SARIMAX exog_cols={sarimax_exog_cols}")
        print("=" * 80)

        dataset_pack = make_dataset(dataset_cfg)
        train_df, valid_df, test_df = (
            dataset_pack["train"],
            dataset_pack["valid"],
            dataset_pack["test"],
        )
        feature_columns = dataset_pack["feature_cols"]

        # Persist bins + feature cols for reproducibility
        bins = dataset_pack.get("bins", None)
        if bins is not None:
            # bins is usually a (ticker x quantiles) DataFrame
            if hasattr(bins, "to_csv"):
                bins.to_csv(variant_dir / "bins.csv")
        (variant_dir / "feature_cols.txt").write_text(
            "\n".join(feature_columns), encoding="utf-8"
        )

        x_train, y_train = build_xy(train_df, feature_columns)
        x_valid, y_valid = build_xy(valid_df, feature_columns)
        x_test, y_test = build_xy(test_df, feature_columns)

        print(
            f"Shapes: train={x_train.shape}, valid={x_valid.shape}, test={x_test.shape}"
        )
        sarimax_features = [c for c in feature_columns if c.startswith("sarimax_")]
        print(
            f"SARIMAX features present: {len(sarimax_features)} -> {sarimax_features}"
        )

        # Models
        logit_model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
            ]
        )

        rf_model = RandomForestClassifier(
            n_estimators=parsed_args.rf_estimators,
            max_depth=None,
            min_samples_leaf=parsed_args.rf_min_leaf,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample",
        )

        metrics_by_model = {}
        metrics_by_model["logit"] = train_eval_one(
            logit_model, x_train, y_train, x_valid, y_valid, x_test, y_test
        )
        metrics_by_model["rf"] = train_eval_one(
            rf_model, x_train, y_train, x_valid, y_valid, x_test, y_test
        )

        save_json(variant_dir / "metrics.json", metrics_by_model)

        # Print key metrics to console
        for name in ["logit", "rf"]:
            va = metrics_by_model[name]["valid"]
            te = metrics_by_model[name]["test"]
            print(f"\n{name.upper()}  VALID acc={va['accuracy']:.4f} macroF1={va['macro_f1']:.4f}")
            print(f"{name.upper()}  TEST  acc={te['accuracy']:.4f} macroF1={te['macro_f1']:.4f}")

            summary_records.append(
                {
                    "variant": variant,
                    "model": name,
                    "valid_acc": va["accuracy"],
                    "valid_macro_f1": va["macro_f1"],
                    "test_acc": te["accuracy"],
                    "test_macro_f1": te["macro_f1"],
                    "n_features": len(feature_columns),
                }
            )

    # Save a compact summary table
    summary_df = pd.DataFrame(summary_records).sort_values(["variant", "model"])
    summary_df.to_csv(output_root / "summary.csv", index=False)
    print("\nSaved summary ->", output_root / "summary.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
