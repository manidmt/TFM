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


def print_model_metrics(name: str, metrics_valid: dict, metrics_test: dict) -> None:
    print(f"\n===== {name} =====")
    print(f"VALID acc={metrics_valid['accuracy']:.4f} macroF1={metrics_valid['macro_f1']:.4f}")
    print(metrics_valid["report"])
    print("Per-class acc (VALID):", metrics_valid["per_class_acc"])

    print(f"\nTEST  acc={metrics_test['accuracy']:.4f} macroF1={metrics_test['macro_f1']:.4f}")
    print(metrics_test["report"])
    print("Per-class acc (TEST):", metrics_test["per_class_acc"])


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
    arg_parser.add_argument("--sarimax_target_col", default="rv_20")

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
        sarimax_target_col=parsed_args.sarimax_target_col,
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

        # def _corr(a, b):
        #     s = np.corrcoef(a, b)[0, 1]
        #     return float(s)

        # def _safe_corr(df, x, y):
        #     if x not in df.columns or y not in df.columns:
        #         return None
        #     tmp = df[[x, y]].dropna()
        #     if len(tmp) < 30:
        #         return None
        #     return _corr(tmp[x].to_numpy(dtype=float), tmp[y].to_numpy(dtype=float))

        # sar_mean = f"sarimax_fcst_mean_h{parsed_args.horizon}"

        # print("\n--- Leakage sanity checks (per split) ---")
        # for name, split in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        #     c1 = _safe_corr(split, sar_mean, "vol_fwd")
        #     c2 = _safe_corr(split, "sarimax_resid", "vol_fwd")
        #     c3 = _safe_corr(split, sar_mean, "rv_20")
        #     c4 = _safe_corr(split, "sarimax_resid", "rv_20")
        #     print(f"{name:5s} corr({sar_mean}, vol_fwd) = {c1}")
        #     print(f"{name:5s} corr(sarimax_resid, vol_fwd) = {c2}")
        #     print(f"{name:5s} corr({sar_mean}, rv_20) = {c3}")
        #     print(f"{name:5s} corr(sarimax_resid, rv_20) = {c4}")

        # h = parsed_args.horizon
        # tmp = test_df.sort_values(["ticker", "date"]).copy()
        # tmp["vol_fwd_plus_h"] = tmp.groupby("ticker")["vol_fwd"].shift(-h)
        # tmp["rv_20_plus_h"] = tmp.groupby("ticker")["rv_20"].shift(-h)

        # print("\n--- Alignment check (TEST) ---")
        # print("corr(sarimax_fcst_mean, vol_fwd[t])   =", _safe_corr(tmp, sar_mean, "vol_fwd"))
        # print("corr(sarimax_fcst_mean, vol_fwd[t+h]) =", _safe_corr(tmp, sar_mean, "vol_fwd_plus_h"))
        # print("corr(sarimax_fcst_mean, rv_20[t])     =", _safe_corr(tmp, sar_mean, "rv_20"))
        # print("corr(sarimax_fcst_mean, rv_20[t+h])   =", _safe_corr(tmp, sar_mean, "rv_20_plus_h"))


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
