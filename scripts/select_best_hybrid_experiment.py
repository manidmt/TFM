'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-23

@description: Run hyperparameter grids for hybrid econometric+tabular classifiers and
select the best experiment per ticker and horizon.
'''

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quant_risk.config import load_yaml, resolve_tickers
from quant_risk.datasets.make_dataset import DatasetConfig, build_xy, make_dataset
from quant_risk.models.metrics import compute_metrics


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _parse_sarimax_order(spec: Any) -> tuple[int, int, int]:
    if isinstance(spec, (list, tuple)) and len(spec) == 3:
        return tuple(int(x) for x in spec)
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if len(parts) == 3:
            return tuple(int(x) for x in parts)
    raise ValueError(f"Invalid sarimax_order={spec}")


def _parse_sarimax_seasonal_order(spec: Any) -> tuple[int, int, int, int]:
    if isinstance(spec, (list, tuple)) and len(spec) == 4:
        return tuple(int(x) for x in spec)
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if len(parts) == 4:
            return tuple(int(x) for x in parts)
    raise ValueError(f"Invalid sarimax_seasonal_order={spec}")


def _variant_to_exog_cols(variant: str, net_liquidity_col: str) -> tuple[str, ...]:
    if variant == "none":
        return ()
    if variant == "nl":
        return (net_liquidity_col,)
    if variant.startswith("exog:"):
        cols = variant.split(":", 1)[1]
        return tuple([col for col in cols.split("|") if col])
    raise ValueError(f"Unknown sarimax_variants option: {variant}")


def build_experiments(model_type: str, grid_config_path: str, default_params: dict[str, Any]) -> list[dict[str, Any]]:
    doc = load_yaml(grid_config_path)
    grid = doc.get("grid", {}) if isinstance(doc, dict) else {}

    if model_type == "garch":
        garch_p_values = _as_list(grid.get("garch_p", [1]))
        garch_q_values = _as_list(grid.get("garch_q", [1]))
        garch_dist_values = _as_list(grid.get("garch_dist", [default_params["garch_dist"]]))
        garch_mean_values = _as_list(grid.get("garch_mean", [default_params["garch_mean"]]))
        garch_vol_values = _as_list(grid.get("garch_vol", [default_params["garch_vol"]]))
        garch_target_values = _as_list(grid.get("garch_target_col", [default_params["garch_target_col"]]))
        garch_scale_values = _as_list(grid.get("garch_scale", [default_params["garch_scale"]]))
        garch_annualize_values = _as_list(grid.get("garch_annualize", [default_params["garch_annualize"]]))
        rf_estimators_values = _as_list(grid.get("rf_estimators", [default_params["rf_estimators"]]))
        rf_min_leaf_values = _as_list(grid.get("rf_min_leaf", [default_params["rf_min_leaf"]]))

        experiments = []
        for idx, (g_p, g_q, g_dist, g_mean, g_vol, g_target, g_scale, g_ann, rf_n, rf_leaf) in enumerate(
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
            experiments.append(
                {
                    "id": f"exp_{idx:04d}",
                    "garch_p": int(g_p),
                    "garch_q": int(g_q),
                    "garch_dist": str(g_dist),
                    "garch_mean": str(g_mean),
                    "garch_vol": str(g_vol),
                    "garch_target_col": str(g_target),
                    "garch_scale": float(g_scale),
                    "garch_annualize": bool(g_ann),
                    "rf_estimators": int(rf_n),
                    "rf_min_leaf": int(rf_leaf),
                }
            )
        return experiments

    sarimax_order_values = _as_list(grid.get("sarimax_order", [default_params["sarimax_order"]]))
    sarimax_seasonal_values = _as_list(
        grid.get("sarimax_seasonal_order", [default_params["sarimax_seasonal_order"]])
    )
    sarimax_trend_values = _as_list(grid.get("sarimax_trend", [default_params["sarimax_trend"]]))
    sarimax_log_transform_values = _as_list(
        grid.get("sarimax_log_transform", [default_params["sarimax_log_transform"]])
    )
    sarimax_variants_values = _as_list(grid.get("sarimax_variants", [default_params["sarimax_variants"]]))
    rf_estimators_values = _as_list(grid.get("rf_estimators", [default_params["rf_estimators"]]))
    rf_min_leaf_values = _as_list(grid.get("rf_min_leaf", [default_params["rf_min_leaf"]]))

    experiments = []
    for idx, (s_order, s_seasonal, s_trend, s_log, s_variant, rf_n, rf_leaf) in enumerate(
        itertools.product(
            sarimax_order_values,
            sarimax_seasonal_values,
            sarimax_trend_values,
            sarimax_log_transform_values,
            sarimax_variants_values,
            rf_estimators_values,
            rf_min_leaf_values,
        )
    ):
        experiments.append(
            {
                "id": f"exp_{idx:04d}",
                "sarimax_order": _parse_sarimax_order(s_order),
                "sarimax_seasonal_order": _parse_sarimax_seasonal_order(s_seasonal),
                "sarimax_trend": str(s_trend),
                "sarimax_log_transform": bool(s_log),
                "sarimax_variant": str(s_variant),
                "rf_estimators": int(rf_n),
                "rf_min_leaf": int(rf_leaf),
            }
        )

    return experiments


def train_eval_valid(model, x_train, y_train, x_valid, y_valid) -> dict:
    model.fit(x_train, y_train)

    pred_valid = model.predict(x_valid)
    metrics_valid = compute_metrics(y_valid, pred_valid)

    return {
        "valid_acc": float(metrics_valid.accuracy),
        "valid_macro_f1": float(metrics_valid.macro_f1),
    }


def eval_test(model, x_test, y_test) -> dict:
    pred_test = model.predict(x_test)
    metrics_test = compute_metrics(y_test, pred_test)
    return {
        "test_acc": float(metrics_test.accuracy),
        "test_macro_f1": float(metrics_test.macro_f1),
    }


def score(metrics: dict[str, float], objective: str) -> tuple[float, float]:
    if objective == "valid_acc":
        return (metrics["valid_acc"], metrics["valid_macro_f1"])
    return (metrics["valid_macro_f1"], metrics["valid_acc"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select best hybrid experiment per ticker and horizon using grid search."
    )
    parser.add_argument("--model", choices=["garch", "sarimax"], default="garch")
    parser.add_argument("--grid_config", required=True)
    parser.add_argument("--config_features", default="config/features.yaml")
    parser.add_argument("--config_sources", default="config/datasources.yaml")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20])
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--pooled", action="store_true")
    parser.add_argument("--objective", choices=["valid_macro_f1", "valid_acc"], default="valid_macro_f1")
    parser.add_argument("--out_json", default="runs/best_hybrid_experiments.json")

    parser.add_argument("--rf_estimators", type=int, default=400)
    parser.add_argument("--rf_min_leaf", type=int, default=5)

    parser.add_argument("--garch_dist", default="normal")
    parser.add_argument("--garch_mean", default="zero")
    parser.add_argument("--garch_vol", default="Garch")
    parser.add_argument("--garch_target_col", default="logret")
    parser.add_argument("--garch_scale", type=float, default=100.0)
    parser.add_argument("--garch_annualize", action="store_true")

    parser.add_argument("--sarimax_target_col", default="rv_20")
    parser.add_argument("--sarimax_order", default="1,0,1")
    parser.add_argument("--sarimax_seasonal_order", default="0,0,0,0")
    parser.add_argument("--sarimax_trend", default="c")
    parser.add_argument(
        "--sarimax_log_transform",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sarimax_variants", default="none")
    parser.add_argument("--net_liquidity_col", default="net_liquidity_diff")

    args = parser.parse_args()

    features_cfg = load_yaml(args.config_features)
    sources_cfg = load_yaml(args.config_sources)
    db_path = sources_cfg["db"]["path"]
    tickers = resolve_tickers(args.tickers, args.config_sources)

    defaults = {
        "rf_estimators": args.rf_estimators,
        "rf_min_leaf": args.rf_min_leaf,
        "garch_dist": args.garch_dist,
        "garch_mean": args.garch_mean,
        "garch_vol": args.garch_vol,
        "garch_target_col": args.garch_target_col,
        "garch_scale": args.garch_scale,
        "garch_annualize": bool(args.garch_annualize),
        "sarimax_order": _parse_sarimax_order(args.sarimax_order),
        "sarimax_seasonal_order": _parse_sarimax_seasonal_order(args.sarimax_seasonal_order),
        "sarimax_trend": args.sarimax_trend,
        "sarimax_log_transform": bool(args.sarimax_log_transform),
        "sarimax_variants": args.sarimax_variants.split(",")[0].strip() if args.sarimax_variants else "none",
    }

    experiments = build_experiments(args.model, args.grid_config, defaults)
    if not experiments:
        raise SystemExit("No experiments generated from grid config.")

    print(f"[info] Running {len(experiments)} experiments for model={args.model}")

    result_payload: dict[str, Any] = {
        "model": args.model,
        "objective": args.objective,
        "grid_config": args.grid_config,
        "best_by_horizon": {},
    }

    for horizon in args.horizons:
        horizon_key = str(horizon)
        result_payload["best_by_horizon"][horizon_key] = {}

        for ticker in tickers:
            print(f"\n[run] Evaluating ticker={ticker} horizon={horizon}")

            best_record: dict[str, Any] | None = None
            skipped = 0

            for experiment in experiments:
                base_cfg = dict(
                    db_path=db_path,
                    tickers=(ticker,),
                    horizon=int(horizon),
                    pooled=bool(args.pooled),
                    train_end=features_cfg["split"]["train_end"],
                    valid_end=features_cfg["split"]["val_end"],
                    regime_bins=features_cfg["targets"]["regime_bins"],
                )

                if args.model == "garch":
                    dataset_cfg = DatasetConfig(
                        **base_cfg,
                        use_garch=True,
                        garch_p=experiment["garch_p"],
                        garch_q=experiment["garch_q"],
                        garch_dist=experiment["garch_dist"],
                        garch_mean=experiment["garch_mean"],
                        garch_vol=experiment["garch_vol"],
                        garch_target_col=experiment["garch_target_col"],
                        garch_scale=experiment["garch_scale"],
                        garch_annualize=experiment["garch_annualize"],
                    )
                else:
                    exog_cols = _variant_to_exog_cols(experiment["sarimax_variant"], args.net_liquidity_col)
                    dataset_cfg = DatasetConfig(
                        **base_cfg,
                        use_sarimax=True,
                        sarimax_order=experiment["sarimax_order"],
                        sarimax_seasonal_order=experiment["sarimax_seasonal_order"],
                        sarimax_trend=experiment["sarimax_trend"],
                        sarimax_log_transform=experiment["sarimax_log_transform"],
                        sarimax_target_col=args.sarimax_target_col,
                        sarimax_exog_cols=exog_cols,
                    )

                try:
                    pack = make_dataset(dataset_cfg)
                except Exception:
                    skipped += 1
                    continue

                train_df = pack["train"]
                valid_df = pack["valid"]
                feature_cols = pack["feature_cols"]

                if train_df.empty or valid_df.empty:
                    skipped += 1
                    continue

                x_train, y_train = build_xy(train_df, feature_cols)
                x_valid, y_valid = build_xy(valid_df, feature_cols)

                logit_model = Pipeline(
                    steps=[
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
                    ]
                )
                rf_model = RandomForestClassifier(
                    n_estimators=experiment["rf_estimators"],
                    max_depth=None,
                    min_samples_leaf=experiment["rf_min_leaf"],
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced_subsample",
                )

                metrics_logit = train_eval_valid(
                    logit_model, x_train, y_train, x_valid, y_valid
                )
                metrics_rf = train_eval_valid(
                    rf_model, x_train, y_train, x_valid, y_valid
                )

                best_model_name = "logit"
                best_model_metrics = metrics_logit
                if score(metrics_rf, args.objective) > score(metrics_logit, args.objective):
                    best_model_name = "rf"
                    best_model_metrics = metrics_rf

                candidate = {
                    "experiment_id": experiment["id"],
                    "params": experiment,
                    "best_model": best_model_name,
                    "metrics": best_model_metrics,
                    "n_features": int(len(feature_cols)),
                }

                if best_record is None or score(candidate["metrics"], args.objective) > score(best_record["metrics"], args.objective):
                    best_record = candidate

            if best_record is None:
                result_payload["best_by_horizon"][horizon_key][ticker] = {
                    "status": "no_valid_experiment",
                    "skipped_experiments": skipped,
                }
                print(f"[warn] No valid experiment for ticker={ticker}, horizon={horizon}")
                continue

            selected_experiment = best_record["params"]
            base_cfg = dict(
                db_path=db_path,
                tickers=(ticker,),
                horizon=int(horizon),
                pooled=bool(args.pooled),
                train_end=features_cfg["split"]["train_end"],
                valid_end=features_cfg["split"]["val_end"],
                regime_bins=features_cfg["targets"]["regime_bins"],
            )
            if args.model == "garch":
                final_cfg = DatasetConfig(
                    **base_cfg,
                    use_garch=True,
                    garch_p=selected_experiment["garch_p"],
                    garch_q=selected_experiment["garch_q"],
                    garch_dist=selected_experiment["garch_dist"],
                    garch_mean=selected_experiment["garch_mean"],
                    garch_vol=selected_experiment["garch_vol"],
                    garch_target_col=selected_experiment["garch_target_col"],
                    garch_scale=selected_experiment["garch_scale"],
                    garch_annualize=selected_experiment["garch_annualize"],
                )
            else:
                exog_cols = _variant_to_exog_cols(
                    selected_experiment["sarimax_variant"], args.net_liquidity_col
                )
                final_cfg = DatasetConfig(
                    **base_cfg,
                    use_sarimax=True,
                    sarimax_order=selected_experiment["sarimax_order"],
                    sarimax_seasonal_order=selected_experiment["sarimax_seasonal_order"],
                    sarimax_trend=selected_experiment["sarimax_trend"],
                    sarimax_log_transform=selected_experiment["sarimax_log_transform"],
                    sarimax_target_col=args.sarimax_target_col,
                    sarimax_exog_cols=exog_cols,
                )

            final_pack = make_dataset(final_cfg)
            final_train = final_pack["train"]
            final_valid = final_pack["valid"]
            final_test = final_pack["test"]
            final_feature_cols = final_pack["feature_cols"]
            if final_train.empty or final_valid.empty or final_test.empty:
                result_payload["best_by_horizon"][horizon_key][ticker] = {
                    "status": "best_no_test_split",
                    "skipped_experiments": skipped,
                }
                print(f"[warn] Best experiment has empty split for ticker={ticker}, horizon={horizon}")
                continue

            x_train, y_train = build_xy(final_train, final_feature_cols)
            x_valid, y_valid = build_xy(final_valid, final_feature_cols)
            x_test, y_test = build_xy(final_test, final_feature_cols)

            if best_record["best_model"] == "logit":
                final_model = Pipeline(
                    steps=[
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
                    ]
                )
            else:
                final_model = RandomForestClassifier(
                    n_estimators=selected_experiment["rf_estimators"],
                    max_depth=None,
                    min_samples_leaf=selected_experiment["rf_min_leaf"],
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced_subsample",
                )

            valid_metrics = train_eval_valid(final_model, x_train, y_train, x_valid, y_valid)
            test_metrics = eval_test(final_model, x_test, y_test)
            best_record["metrics"] = {**valid_metrics, **test_metrics}
            best_record["n_features"] = int(len(final_feature_cols))
            best_record["status"] = "ok"
            best_record["skipped_experiments"] = skipped
            result_payload["best_by_horizon"][horizon_key][ticker] = best_record

            print(
                f"[best] ticker={ticker} h={horizon} exp={best_record['experiment_id']} "
                f"model={best_record['best_model']} "
                f"valid_macro_f1={best_record['metrics']['valid_macro_f1']:.4f} "
                f"valid_acc={best_record['metrics']['valid_acc']:.4f}"
            )

    save_json(Path(args.out_json), result_payload)
    print(f"\nSaved best experiments JSON -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
