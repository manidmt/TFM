"""
Experiment sweep runner using Optuna + MLflow.

Usage:
    poetry run python scripts/sweep_runner.py --config config/sweep_config.yaml
    poetry run python scripts/sweep_runner.py --config config/sweep_config.yaml --n_trials 2 --assets "^GSPC"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import optuna
import yaml

# Suppress Optuna info-level chatter
optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = Path(__file__).parent.parent


def load_sweep_config(path: str) -> dict[str, Any]:
    """Load and return the sweep YAML config."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sweep config not found: {path}")
    with open(p) as f:
        return yaml.safe_load(f)


def build_trial_args(
    trial: Any,
    search_space: dict[str, Any],
    asset: str,
    base_args: list[str],
) -> list[str]:
    """Sample Optuna params and return CLI args for walk_forward_chain_tab.py."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "float":
            log = bool(spec.get("log", False))
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=log)

    args: list[str] = list(base_args)

    # Always-present flags
    args += ["--tabular_model", str(params["tabular_model"])]
    args += ["--chain_variants_config", str(params["chain_variants_config"])]

    # store_true flags
    if str(params.get("use_blend", "false")) == "true":
        args.append("--use_blend")
    if str(params.get("use_gkg_change_detector", "false")) == "true":
        args.append("--use_gkg_change_detector")

    # Model-conditional hyperparams
    if str(params["tabular_model"]) == "tabpfn":
        args += ["--tabpfn_n_estimators", str(params["tabpfn_n_estimators"])]
        args += ["--tabpfn_softmax_temperature", str(params["tabpfn_softmax_temperature"])]
    elif str(params["tabular_model"]) == "xgb":
        args += ["--xgb_max_depth", str(params["xgb_max_depth"])]
        args += ["--xgb_learning_rate", str(params["xgb_learning_rate"])]

    # GKG-conditional flags
    if str(params.get("use_gkg_change_detector", "false")) == "true":
        args += ["--gkg_change_model", str(params["gkg_change_model"])]
        if str(params["gkg_change_model"]) == "tabpfn":
            args += ["--gkg_tabpfn_n_estimators", str(params["gkg_tabpfn_n_estimators"])]

    return args


def run_trial_subprocess(cli_args: list[str], repo_root: Path) -> int:
    """Run walk_forward_chain_tab.py and return exit code."""
    cmd = ["poetry", "run", "python", "scripts/walk_forward_chain_tab.py"] + cli_args
    result = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return result.returncode


def read_robust_score_from_mlflow(
    experiment_name: str,
    asset: str,
    trial_number: int,
    study_name: str,
) -> float:
    """Query MLflow for the child run's robust_score for this trial."""
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=(
                f"tags.asset = '{asset}' "
                f"AND tags.sweep_trial = '{trial_number}' "
                f"AND tags.optuna_study = '{study_name}'"
            ),
            order_by=["start_time DESC"],
            max_results=1,
        )
    except Exception as exc:
        warnings.warn(f"[sweep] MLflow query failed for trial {trial_number}: {exc}")
        return float("-inf")

    if runs.empty:
        warnings.warn(f"[sweep] No MLflow run found for trial {trial_number}, asset={asset}")
        return float("-inf")

    score = runs.iloc[0].get("metrics.robust_score")
    if score is None or (isinstance(score, float) and not math.isfinite(score)):
        warnings.warn(f"[sweep] robust_score missing or non-finite for trial {trial_number}")
        return float("-inf")
    return float(score)


def make_objective(
    asset: str,
    search_space: dict[str, Any],
    base_args: list[str],
    experiment_name: str,
    study_name: str,
    strict: bool = False,
) -> Any:
    """Return the Optuna objective closure for one asset."""

    def objective(trial: Any) -> float:
        trial_args = build_trial_args(trial, search_space, asset, base_args)
        mlflow_args = [
            "--use_mlflow",
            "--mlflow_experiment", experiment_name,
            "--mlflow_tags_json", json.dumps({
                "sweep_trial": str(trial.number),
                "optuna_study": study_name,
            }),
            "--asset", asset,
        ]
        all_args = mlflow_args + trial_args
        print(f"\n[sweep] Trial {trial.number} | asset={asset} | args: {' '.join(all_args)}")

        exit_code = run_trial_subprocess(all_args, REPO_ROOT)
        if exit_code != 0:
            msg = f"Trial {trial.number} subprocess failed (exit {exit_code})"
            if strict:
                raise RuntimeError(msg)
            warnings.warn(f"[sweep] {msg}")
            return float("-inf")

        return read_robust_score_from_mlflow(experiment_name, asset, trial.number, study_name)

    return objective


def log_sweep_summary(
    experiment_name: str,
    asset: str,
    study: Any,
    n_failed: int,
) -> None:
    """Log best params and study summary to a sweep summary MLflow run."""
    mlflow.set_experiment(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sweep__{asset}__{timestamp}"

    successful = [t for t in study.trials if t.value is not None and math.isfinite(t.value)]
    n_successful = len(successful)
    best_score = study.best_value if n_successful > 0 else float("-inf")
    best_params = study.best_params if n_successful > 0 else {}

    summary = {
        "asset": asset,
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "n_successful": n_successful,
        "n_failed": n_failed,
        "best_score": best_score,
        "best_params": best_params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ],
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "asset": asset,
            "optuna_study_name": study.study_name,
            "n_trials": str(len(study.trials)),
        })
        mlflow.log_params({k: str(v) for k, v in best_params.items()})
        mlflow.log_metrics({
            "best_robust_score": best_score,
            "n_successful_trials": float(n_successful),
            "n_failed_trials": float(n_failed),
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "optuna_study_summary.json")
            with open(fpath, "w") as f:
                json.dump(summary, f, indent=2)
            mlflow.log_artifact(fpath)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna sweep runner over walk_forward_chain_tab.py")
    parser.add_argument("--config", default="config/sweep_config.yaml", help="Path to sweep_config.yaml")
    parser.add_argument("--n_trials", type=int, default=None, help="Override n_trials from config")
    parser.add_argument("--assets", nargs="+", default=None, help="Override assets from config")
    parser.add_argument("--sweep_strict", action="store_true", help="Abort on any trial failure")
    args = parser.parse_args()

    cfg = load_sweep_config(args.config)
    sweep_cfg = cfg["sweep"]
    search_space = cfg["search_space"]

    n_trials = args.n_trials if args.n_trials is not None else int(sweep_cfg["n_trials"])
    assets = args.assets if args.assets is not None else list(sweep_cfg["assets"])
    experiment_name = str(sweep_cfg["mlflow_experiment"])

    mlflow.set_tracking_uri("file:./mlruns")

    for asset in assets:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Asset name sanitised for use in study_name (^ and - are invalid in some backends)
        asset_safe = asset.replace("^", "").replace("-", "_")
        study_name = f"sweep_{asset_safe}_{timestamp}"

        print(f"\n[sweep] Starting Optuna study for asset={asset} | study={study_name} | n_trials={n_trials}")
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        n_failed = 0
        objective = make_objective(
            asset=asset,
            search_space=search_space,
            base_args=[],
            experiment_name=experiment_name,
            study_name=study_name,
            strict=args.sweep_strict,
        )

        def objective_with_count(trial: Any) -> float:
            nonlocal n_failed
            score = objective(trial)
            if not math.isfinite(score):
                n_failed += 1
            return score

        study.optimize(objective_with_count, n_trials=n_trials)

        best = study.best_params if any(math.isfinite(t.value or float("-inf")) for t in study.trials) else {}
        print(f"\n[sweep] asset={asset} | best_robust_score={study.best_value:.4f} | best_params={best}")

        log_sweep_summary(experiment_name, asset, study, n_failed)

    print("\n[sweep] Done.")


if __name__ == "__main__":
    main()
