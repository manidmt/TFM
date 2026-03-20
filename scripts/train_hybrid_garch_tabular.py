'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-23

@description: Train hybrid volatility-regime classifiers with GARCH + tabular features.
Runs multiple GARCH variants and persists metrics/artifacts.
'''

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from quant_risk.config import load_yaml, resolve_tickers
from quant_risk.datasets.make_dataset import DatasetConfig, build_xy, make_dataset
from quant_risk.models.metrics import compute_metrics, per_class_accuracy


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
        "garch_variants": "--garch_variants",
        "garch_dist": "--garch_dist",
        "garch_mean": "--garch_mean",
        "garch_vol": "--garch_vol",
        "garch_target_col": "--garch_target_col",
        "garch_scale": "--garch_scale",
        "garch_annualize": "--garch_annualize",
        "rf_estimators": "--rf_estimators",
        "rf_min_leaf": "--rf_min_leaf",
        "outdir": "--outdir",
    }

    for key, flag in mapping.items():
        if key in preset and not _flag_provided(flag):
            setattr(parsed_args, key, preset[key])

    return parsed_args


def _parse_garch_variants(spec: str) -> list[tuple[str, int, int]]:
    variants: list[tuple[str, int, int]] = []
    raw = [v.strip() for v in spec.split(",") if v.strip()]

    for variant in raw:
        if variant in {"11", "1-1"}:
            variants.append((variant, 1, 1))
            continue

        if variant.startswith("pq:"):
            value = variant.split(":", 1)[1]
            if "-" not in value:
                raise SystemExit(
                    f"Invalid variant '{variant}'. Use pq:P-Q, e.g. pq:1-2"
                )
            p_text, q_text = value.split("-", 1)
            p = int(p_text)
            q = int(q_text)
            if p <= 0 or q <= 0:
                raise SystemExit(f"Invalid GARCH orders in '{variant}'. p and q must be >= 1")
            variants.append((variant, p, q))
            continue

        if "-" in variant:
            p_text, q_text = variant.split("-", 1)
            p = int(p_text)
            q = int(q_text)
            if p <= 0 or q <= 0:
                raise SystemExit(f"Invalid GARCH orders in '{variant}'. p and q must be >= 1")
            variants.append((variant, p, q))
            continue

        raise SystemExit(
            f"Unknown variant '{variant}'. Use 11, 1-1, 1-2,2-1 or pq:1-2"
        )

    if not variants:
        raise SystemExit("No variants provided.")

    return variants


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _build_garch_experiments(parsed_args: argparse.Namespace) -> list[dict[str, Any]]:
    if not parsed_args.grid_config:
        variants = _parse_garch_variants(str(parsed_args.garch_variants))
        return [
            {
                "variant_name": variant_name,
                "garch_p": int(p),
                "garch_q": int(q),
                "garch_dist": parsed_args.garch_dist,
                "garch_mean": parsed_args.garch_mean,
                "garch_vol": parsed_args.garch_vol,
                "garch_target_col": parsed_args.garch_target_col,
                "garch_scale": float(parsed_args.garch_scale),
                "garch_annualize": bool(parsed_args.garch_annualize),
                "rf_estimators": int(parsed_args.rf_estimators),
                "rf_min_leaf": int(parsed_args.rf_min_leaf),
            }
            for variant_name, p, q in variants
        ]

    grid_doc = load_yaml(parsed_args.grid_config)
    grid = grid_doc.get("grid", {}) if isinstance(grid_doc, dict) else {}

    garch_p_values = _as_list(grid.get("garch_p", [1]))
    garch_q_values = _as_list(grid.get("garch_q", [1]))
    garch_dist_values = _as_list(grid.get("garch_dist", [parsed_args.garch_dist]))
    garch_mean_values = _as_list(grid.get("garch_mean", [parsed_args.garch_mean]))
    garch_vol_values = _as_list(grid.get("garch_vol", [parsed_args.garch_vol]))
    garch_target_values = _as_list(grid.get("garch_target_col", [parsed_args.garch_target_col]))
    garch_scale_values = _as_list(grid.get("garch_scale", [float(parsed_args.garch_scale)]))
    garch_annualize_values = _as_list(grid.get("garch_annualize", [bool(parsed_args.garch_annualize)]))
    rf_estimators_values = _as_list(grid.get("rf_estimators", [int(parsed_args.rf_estimators)]))
    rf_min_leaf_values = _as_list(grid.get("rf_min_leaf", [int(parsed_args.rf_min_leaf)]))

    experiments: list[dict[str, Any]] = []
    for idx, (p, q, dist, mean, vol, target_col, scale, annualize, rf_n, rf_leaf) in enumerate(
        itertools.product(
            garch_p_values,
            garch_q_values,
            garch_dist_values,
            garch_mean_values,
            garch_vol_values,
            garch_target_values,
            garch_scale_values,
            garch_annualize_values,
            rf_estimators_values,
            rf_min_leaf_values,
        )
    ):
        p_i = int(p)
        q_i = int(q)
        if p_i <= 0 or q_i <= 0:
            raise SystemExit(f"Invalid grid config p/q combination: p={p_i}, q={q_i}")

        variant_name = (
            f"grid{idx:03d}_p{p_i}_q{q_i}_dist-{dist}_mean-{mean}"
            f"_sc-{float(scale):.3g}_ann-{int(bool(annualize))}"
            f"_rfn-{int(rf_n)}_leaf-{int(rf_leaf)}"
        )
        experiments.append(
            {
                "variant_name": variant_name,
                "garch_p": p_i,
                "garch_q": q_i,
                "garch_dist": str(dist),
                "garch_mean": str(mean),
                "garch_vol": str(vol),
                "garch_target_col": str(target_col),
                "garch_scale": float(scale),
                "garch_annualize": bool(annualize),
                "rf_estimators": int(rf_n),
                "rf_min_leaf": int(rf_leaf),
            }
        )

    if not experiments:
        raise SystemExit(f"No GARCH experiments generated from grid config: {parsed_args.grid_config}")

    print(f"[grid] Generated {len(experiments)} GARCH experiments from {parsed_args.grid_config}")
    return experiments


def main() -> int:
    arg_parser = argparse.ArgumentParser(
        description="Train hybrid GARCH + tabular models for regime classification."
    )
    arg_parser.add_argument("--config_features", default="config/features.yaml")
    arg_parser.add_argument("--config_sources", default="config/datasources.yaml")
    arg_parser.add_argument("--horizon", type=int, default=20, choices=[5, 20])
    arg_parser.add_argument("--tickers", nargs="+", default=None)
    arg_parser.add_argument("--pooled", action="store_true")

    arg_parser.add_argument("--preset", default=None)
    arg_parser.add_argument("--preset_file", default="config/hybrid_garch_presets.yaml")
    arg_parser.add_argument("--grid_config", default=None)

    arg_parser.add_argument("--garch_variants", default="11")
    arg_parser.add_argument("--garch_dist", default="normal")
    arg_parser.add_argument("--garch_mean", default="zero")
    arg_parser.add_argument("--garch_vol", default="Garch")
    arg_parser.add_argument("--garch_target_col", default="logret")
    arg_parser.add_argument("--garch_scale", type=float, default=100.0)
    arg_parser.add_argument("--garch_annualize", action="store_true")

    arg_parser.add_argument("--rf_estimators", type=int, default=400)
    arg_parser.add_argument("--rf_min_leaf", type=int, default=5)
    arg_parser.add_argument("--outdir", default="runs/hybrid_garch")

    parsed_args = arg_parser.parse_args()
    parsed_args = _apply_preset_if_requested(parsed_args)

    features_config_dict = load_yaml(parsed_args.config_features)
    sources_config_dict = load_yaml(parsed_args.config_sources)
    database_path = sources_config_dict["db"]["path"]
    tickers = resolve_tickers(parsed_args.tickers, parsed_args.config_sources)

    output_root = Path(parsed_args.outdir) / f"h{parsed_args.horizon}"
    ensure_dir(output_root)

    experiments = _build_garch_experiments(parsed_args)

    base_kwargs = dict(
        db_path=database_path,
        tickers=tickers,
        horizon=parsed_args.horizon,
        pooled=bool(parsed_args.pooled),
        train_end=features_config_dict["split"]["train_end"],
        valid_end=features_config_dict["split"]["val_end"],
        regime_bins=features_config_dict["targets"]["regime_bins"],
        use_garch=True,
    )

    summary_records = []

    for exp in experiments:
        variant_name = exp["variant_name"]
        variant_dir = output_root / f"variant_{variant_name}"
        ensure_dir(variant_dir)

        dataset_cfg = DatasetConfig(
            **base_kwargs,
            garch_p=exp["garch_p"],
            garch_q=exp["garch_q"],
            garch_dist=exp["garch_dist"],
            garch_mean=exp["garch_mean"],
            garch_vol=exp["garch_vol"],
            garch_target_col=exp["garch_target_col"],
            garch_annualize=exp["garch_annualize"],
            garch_scale=exp["garch_scale"],
        )

        save_json(
            variant_dir / "config.json",
            {
                "variant": variant_name,
                "dataset": {
                    "db_path": database_path,
                    "tickers": list(tickers),
                    "horizon": int(parsed_args.horizon),
                    "pooled": bool(parsed_args.pooled),
                    "train_end": features_config_dict["split"]["train_end"],
                    "valid_end": features_config_dict["split"]["val_end"],
                    "regime_bins": int(features_config_dict["targets"]["regime_bins"]),
                },
                "garch": {
                    "p": int(exp["garch_p"]),
                    "q": int(exp["garch_q"]),
                    "dist": exp["garch_dist"],
                    "mean": exp["garch_mean"],
                    "vol": exp["garch_vol"],
                    "target_col": exp["garch_target_col"],
                    "scale": float(exp["garch_scale"]),
                    "annualize": bool(exp["garch_annualize"]),
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
            f"{variant_name} | GARCH(p={exp['garch_p']}, q={exp['garch_q']}) "
            f"dist={exp['garch_dist']} mean={exp['garch_mean']} "
            f"scale={exp['garch_scale']} annualize={exp['garch_annualize']}"
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
        garch_features = [c for c in feature_columns if c.startswith("garch_")]
        print(
            f"GARCH features present: {len(garch_features)} -> {garch_features}"
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
                    "garch_p": exp["garch_p"],
                    "garch_q": exp["garch_q"],
                    "garch_dist": exp["garch_dist"],
                    "garch_mean": exp["garch_mean"],
                    "garch_vol": exp["garch_vol"],
                    "garch_target_col": exp["garch_target_col"],
                    "garch_scale": exp["garch_scale"],
                    "garch_annualize": bool(exp["garch_annualize"]),
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
