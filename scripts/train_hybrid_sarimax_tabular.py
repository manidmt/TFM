'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-10

@description: Train hybrid volatility-regime classifiers with SARIMAX + tabular features.
Runs multiple SARIMAX exog variants and persists metrics/artifacts.
'''

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import yaml
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


def print_model_metrics(name: str, metrics_valid: dict, metrics_test: dict) -> None:
    print(f"\n===== {name} =====")
    print(f"VALID acc={metrics_valid['accuracy']:.4f} macroF1={metrics_valid['macro_f1']:.4f}")
    print(metrics_valid["report"])
    print("Per-class acc (VALID):", metrics_valid["per_class_acc"])

    print(f"\nTEST  acc={metrics_test['accuracy']:.4f} macroF1={metrics_test['macro_f1']:.4f}")
    print(metrics_test["report"])
    print("Per-class acc (TEST):", metrics_test["per_class_acc"])


def _flag_provided(flag: str) -> bool:
    return flag in sys.argv


def _apply_preset_if_requested(parsed_args: argparse.Namespace) -> argparse.Namespace:
    if not parsed_args.preset:
        return parsed_args

    preset_doc = load_yaml(parsed_args.preset_file)
    presets = preset_doc.get("presets", {}) if isinstance(preset_doc, dict) else {}
    if parsed_args.preset not in presets:
        raise SystemExit(
            f"Preset '{parsed_args.preset}' not found in {parsed_args.preset_file}. "
            f"Available: {list(presets.keys())}"
        )

    preset = presets[parsed_args.preset]
    mapping = {
        "config_features": "--config_features",
        "config_sources": "--config_sources",
        "horizon": "--horizon",
        "tickers": "--tickers",
        "pooled": "--pooled",
        "sarimax_variants": "--sarimax_variants",
        "net_liquidity_col": "--net_liquidity_col",
        "sarimax_target_col": "--sarimax_target_col",
        "sarimax_order": "--sarimax_order",
        "sarimax_seasonal_order": "--sarimax_seasonal_order",
        "sarimax_trend": "--sarimax_trend",
        "sarimax_log_transform": "--sarimax_log_transform",
        "rf_estimators": "--rf_estimators",
        "rf_min_leaf": "--rf_min_leaf",
        "outdir": "--outdir",
    }

    for key, flag in mapping.items():
        if key in preset and not _flag_provided(flag):
            setattr(parsed_args, key, preset[key])

    return parsed_args


def _parse_sarimax_order(spec: Any) -> tuple[int, int, int]:
    if isinstance(spec, (list, tuple)) and len(spec) == 3:
        return tuple(int(x) for x in spec)
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if len(parts) == 3:
            return tuple(int(x) for x in parts)
    raise SystemExit(f"Invalid sarimax_order '{spec}'. Use 'p,d,q' or [p,d,q].")


def _parse_sarimax_seasonal_order(spec: Any) -> tuple[int, int, int, int]:
    if isinstance(spec, (list, tuple)) and len(spec) == 4:
        return tuple(int(x) for x in spec)
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if len(parts) == 4:
            return tuple(int(x) for x in parts)
    raise SystemExit(
        f"Invalid sarimax_seasonal_order '{spec}'. Use 'P,D,Q,s' or [P,D,Q,s]."
    )


def _variant_to_exog_cols(variant: str, net_liquidity_col: str) -> tuple[str, ...]:
    if variant == "none":
        return ()
    if variant == "nl":
        return (net_liquidity_col,)
    if variant.startswith("exog:"):
        cols = variant.split(":", 1)[1]
        return tuple([c for c in cols.split("|") if c])
    raise SystemExit(f"Unknown variant '{variant}'. Use none,nl or exog:col1|col2")


def _build_sarimax_experiments(parsed_args: argparse.Namespace) -> list[dict[str, Any]]:
    if not parsed_args.grid_config:
        variants = [v.strip() for v in str(parsed_args.sarimax_variants).split(",") if v.strip()]
        if not variants:
            raise SystemExit("No variants provided.")
        return [
            {
                "variant_name": variant,
                "sarimax_order": _parse_sarimax_order(parsed_args.sarimax_order),
                "sarimax_seasonal_order": _parse_sarimax_seasonal_order(parsed_args.sarimax_seasonal_order),
                "sarimax_trend": parsed_args.sarimax_trend,
                "sarimax_log_transform": bool(parsed_args.sarimax_log_transform),
                "sarimax_exog_cols": _variant_to_exog_cols(variant, parsed_args.net_liquidity_col),
                "rf_estimators": int(parsed_args.rf_estimators),
                "rf_min_leaf": int(parsed_args.rf_min_leaf),
            }
            for variant in variants
        ]

    grid_doc = load_yaml(parsed_args.grid_config)
    grid = grid_doc.get("grid", {}) if isinstance(grid_doc, dict) else {}

    orders = grid.get("sarimax_order", [list(_parse_sarimax_order(parsed_args.sarimax_order))])
    seasonal_orders = grid.get(
        "sarimax_seasonal_order",
        [list(_parse_sarimax_seasonal_order(parsed_args.sarimax_seasonal_order))],
    )
    trends = grid.get("sarimax_trend", [parsed_args.sarimax_trend])
    log_transforms = grid.get("sarimax_log_transform", [bool(parsed_args.sarimax_log_transform)])
    variants = grid.get("sarimax_variants", [v.strip() for v in str(parsed_args.sarimax_variants).split(",") if v.strip()])
    rf_estimators_values = grid.get("rf_estimators", [int(parsed_args.rf_estimators)])
    rf_min_leaf_values = grid.get("rf_min_leaf", [int(parsed_args.rf_min_leaf)])

    experiments: list[dict[str, Any]] = []
    for idx, (order, seasonal, trend, log_tf, variant, rf_n, rf_leaf) in enumerate(
        itertools.product(
            orders,
            seasonal_orders,
            trends,
            log_transforms,
            variants,
            rf_estimators_values,
            rf_min_leaf_values,
        )
    ):
        order_t = _parse_sarimax_order(order)
        seasonal_t = _parse_sarimax_seasonal_order(seasonal)
        exog_cols = _variant_to_exog_cols(str(variant), parsed_args.net_liquidity_col)
        variant_name = (
            f"grid{idx:03d}_v-{variant}_ord-{'-'.join(map(str, order_t))}"
            f"_sord-{'-'.join(map(str, seasonal_t))}_tr-{trend}_log-{int(bool(log_tf))}"
            f"_rfn-{int(rf_n)}_leaf-{int(rf_leaf)}"
        )
        experiments.append(
            {
                "variant_name": variant_name,
                "sarimax_order": order_t,
                "sarimax_seasonal_order": seasonal_t,
                "sarimax_trend": trend,
                "sarimax_log_transform": bool(log_tf),
                "sarimax_exog_cols": exog_cols,
                "rf_estimators": int(rf_n),
                "rf_min_leaf": int(rf_leaf),
            }
        )

    if not experiments:
        raise SystemExit(f"No SARIMAX experiments generated from grid config: {parsed_args.grid_config}")

    print(f"[grid] Generated {len(experiments)} SARIMAX experiments from {parsed_args.grid_config}")
    return experiments


def main() -> int:
    arg_parser = argparse.ArgumentParser(
        description="Train hybrid SARIMAX + tabular models for regime classification."
    )
    arg_parser.add_argument("--config_features", default="config/features.yaml")
    arg_parser.add_argument("--config_sources", default="config/datasources.yaml")
    arg_parser.add_argument("--horizon", type=int, default=20, choices=[5, 20])
    arg_parser.add_argument("--tickers", nargs="+", default=["^GSPC", "BTC-USD", "TLT"])
    arg_parser.add_argument("--pooled", action="store_true")

    arg_parser.add_argument("--preset", default=None)
    arg_parser.add_argument("--preset_file", default="config/hybrid_sarimax_presets.yaml")
    arg_parser.add_argument("--grid_config", default=None)

    arg_parser.add_argument("--sarimax_variants", default="none,nl")
    arg_parser.add_argument("--net_liquidity_col", default="net_liquidity_diff")
    arg_parser.add_argument("--sarimax_target_col", default="rv_20")
    arg_parser.add_argument("--sarimax_order", default="1,0,1")
    arg_parser.add_argument("--sarimax_seasonal_order", default="0,0,0,0")
    arg_parser.add_argument("--sarimax_trend", default="c")
    arg_parser.add_argument("--sarimax_log_transform", action="store_true", default=True)

    arg_parser.add_argument("--rf_estimators", type=int, default=400)
    arg_parser.add_argument("--rf_min_leaf", type=int, default=5)
    arg_parser.add_argument("--outdir", default="runs/hybrid_sarimax")

    parsed_args = arg_parser.parse_args()
    parsed_args = _apply_preset_if_requested(parsed_args)

    features_config_dict = load_yaml(parsed_args.config_features)
    sources_config_dict = load_yaml(parsed_args.config_sources)
    database_path = sources_config_dict["db"]["path"]

    output_root = Path(parsed_args.outdir) / f"h{parsed_args.horizon}"
    ensure_dir(output_root)

    experiments = _build_sarimax_experiments(parsed_args)

    base_kwargs = dict(
        db_path=database_path,
        tickers=tuple(parsed_args.tickers),
        horizon=parsed_args.horizon,
        pooled=bool(parsed_args.pooled),
        train_end=features_config_dict["split"]["train_end"],
        valid_end=features_config_dict["split"]["val_end"],
        regime_bins=features_config_dict["targets"]["regime_bins"],
        use_sarimax=True,
        sarimax_target_col=parsed_args.sarimax_target_col,
    )

    summary_records = []

    for exp in experiments:
        variant_name = exp["variant_name"]
        sarimax_exog_cols = exp["sarimax_exog_cols"]

        variant_dir = output_root / f"variant_{variant_name}"
        ensure_dir(variant_dir)

        dataset_cfg = DatasetConfig(
            **base_kwargs,
            sarimax_order=exp["sarimax_order"],
            sarimax_seasonal_order=exp["sarimax_seasonal_order"],
            sarimax_trend=exp["sarimax_trend"],
            sarimax_log_transform=exp["sarimax_log_transform"],
            sarimax_exog_cols=tuple(sarimax_exog_cols),
        )

        save_json(
            variant_dir / "config.json",
            {
                "variant": variant_name,
                "dataset": {
                    "db_path": database_path,
                    "tickers": list(parsed_args.tickers),
                    "horizon": int(parsed_args.horizon),
                    "pooled": bool(parsed_args.pooled),
                    "train_end": features_config_dict["split"]["train_end"],
                    "valid_end": features_config_dict["split"]["val_end"],
                    "regime_bins": int(features_config_dict["targets"]["regime_bins"]),
                },
                "sarimax": {
                    "target_col": parsed_args.sarimax_target_col,
                    "order": list(exp["sarimax_order"]),
                    "seasonal_order": list(exp["sarimax_seasonal_order"]),
                    "trend": exp["sarimax_trend"],
                    "log_transform": bool(exp["sarimax_log_transform"]),
                    "exog_cols": list(exp["sarimax_exog_cols"]),
                },
                "tabular_model": {
                    "rf_estimators": int(exp["rf_estimators"]),
                    "rf_min_leaf": int(exp["rf_min_leaf"]),
                    "random_state": 42,
                },
            },
        )

        print("\n" + "=" * 80)
        print(
            "Variant="
            f"{variant_name} | SARIMAX exog_cols={sarimax_exog_cols} "
            f"order={exp['sarimax_order']} seasonal={exp['sarimax_seasonal_order']} "
            f"trend={exp['sarimax_trend']} log={exp['sarimax_log_transform']}"
        )
        print("=" * 80)

        dataset_pack = make_dataset(dataset_cfg)
        train_df, valid_df, test_df = (
            dataset_pack["train"],
            dataset_pack["valid"],
            dataset_pack["test"],
        )
        feature_columns = dataset_pack["feature_cols"]

        bins = dataset_pack.get("bins", None)
        if bins is not None and hasattr(bins, "to_csv"):
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

        logit_model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
            ]
        )

        rf_model = RandomForestClassifier(
            n_estimators=exp["rf_estimators"],
            max_depth=None,
            min_samples_leaf=exp["rf_min_leaf"],
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

        display_names = {
            "logit": "LOGIT (multinomial)",
            "rf": "RandomForest",
        }
        for name in ["logit", "rf"]:
            va = metrics_by_model[name]["valid"]
            te = metrics_by_model[name]["test"]
            print_model_metrics(display_names[name], va, te)

            summary_records.append(
                {
                    "variant": variant_name,
                    "model": name,
                    "valid_acc": va["accuracy"],
                    "valid_macro_f1": va["macro_f1"],
                    "test_acc": te["accuracy"],
                    "test_macro_f1": te["macro_f1"],
                    "n_features": len(feature_columns),
                    "sarimax_order": list(exp["sarimax_order"]),
                    "sarimax_seasonal_order": list(exp["sarimax_seasonal_order"]),
                    "sarimax_trend": exp["sarimax_trend"],
                    "sarimax_log_transform": bool(exp["sarimax_log_transform"]),
                    "sarimax_exog_cols": "|".join(exp["sarimax_exog_cols"]),
                    "rf_estimators": exp["rf_estimators"],
                    "rf_min_leaf": exp["rf_min_leaf"],
                }
            )

    summary_df = pd.DataFrame(summary_records).sort_values(["variant", "model"])
    summary_df.to_csv(output_root / "summary.csv", index=False)
    print("\nSaved summary ->", output_root / "summary.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
