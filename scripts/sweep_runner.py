"""
Experiment sweep runner using Optuna + MLflow.

Usage:
    poetry run python scripts/sweep_runner.py --config config/sweep_config.yaml
    poetry run python scripts/sweep_runner.py --config config/sweep_config.yaml --n_trials 2 --assets "^GSPC"

Resume an interrupted sweep (study persisted in SQLite):
    poetry run python scripts/sweep_runner.py \\
        --config config/sweep_config.yaml \\
        --study_storage sqlite:///optuna_studies.db \\
        --resume

Warm-start h5 sweep from h20 best params:
    poetry run python scripts/sweep_runner.py \\
        --config config/sweep_config.yaml \\
        --study_storage sqlite:///optuna_studies_h5.db \\
        --enqueue_best_from walk_fordward_sweep_adv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
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
    use_blend = str(params.get("use_blend", "false")) == "true"
    use_gkg_change_gate = str(params.get("use_gkg_change_gate", "false")) == "true"
    use_gkg_change_alpha = str(params.get("use_gkg_change_alpha", "false")) == "true"
    # Gate/alpha only make sense if the detector is enabled; force coherence here so
    # Optuna does not waste trials on invalid CLI combinations.
    use_gkg_change_detector = (
        str(params.get("use_gkg_change_detector", "false")) == "true"
        or use_gkg_change_gate
        or use_gkg_change_alpha
    )

    # Always-present flags
    args += ["--tabular_model", str(params["tabular_model"])]
    args += ["--chain_variants_config", str(params["chain_variants_config"])]

    # store_true flags
    if use_blend:
        args.append("--use_blend")
    if use_gkg_change_detector:
        args.append("--use_gkg_change_detector")
    if use_gkg_change_gate:
        args.append("--use_gkg_change_gate")
    if use_gkg_change_alpha:
        args.append("--use_gkg_change_alpha")

    # Model-conditional hyperparams
    if str(params["tabular_model"]) == "tabpfn":
        args += ["--tabpfn_n_estimators", str(params["tabpfn_n_estimators"])]
        args += ["--tabpfn_softmax_temperature", str(params["tabpfn_softmax_temperature"])]
    elif str(params["tabular_model"]) == "xgb":
        args += ["--xgb_max_depth", str(params["xgb_max_depth"])]
        args += ["--xgb_learning_rate", str(params["xgb_learning_rate"])]

    # GKG-conditional flags
    if use_gkg_change_detector:
        args += ["--gkg_change_model", str(params["gkg_change_model"])]
        if use_gkg_change_gate and "gkg_change_gate_thresholds" in params:
            args += [
                "--gkg_change_gate_thresholds",
                str(params["gkg_change_gate_thresholds"]),
            ]
        if use_gkg_change_alpha and "gkg_change_alpha_weights" in params:
            args += [
                "--gkg_change_alpha_weights",
                str(params["gkg_change_alpha_weights"]),
            ]
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


def get_best_params_from_experiment(experiment_name: str, asset: str) -> dict[str, Any] | None:
    """Query MLflow for the best sweep_summary params for *asset* in *experiment_name*.

    Returns the best_params dict logged by log_sweep_summary, or None if not found.
    The params are stored as MLflow params (string values); we return them as-is so
    study.enqueue_trial() can coerce them to the right types via the search_space.
    """
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=(
                f"tags.asset = '{asset}' "
                f"AND tags.run_role = 'sweep_summary'"
            ),
            order_by=["metrics.best_robust_score DESC"],
            max_results=1,
        )
    except Exception as exc:
        warnings.warn(f"[sweep] enqueue_best_from: MLflow query failed for asset={asset}: {exc}")
        return None

    if runs.empty:
        warnings.warn(
            f"[sweep] enqueue_best_from: no sweep_summary run found for asset={asset} "
            f"in experiment '{experiment_name}'"
        )
        return None

    row = runs.iloc[0]
    params: dict[str, Any] = {}
    for col in row.index:
        if col.startswith("params."):
            key = col[len("params."):]
            val = row[col]
            if val is not None and str(val) not in ("nan", "None", ""):
                params[key] = val

    if not params:
        warnings.warn(
            f"[sweep] enqueue_best_from: sweep_summary found for asset={asset} but has no params"
        )
        return None

    return params


def _coerce_enqueued_params(
    raw: dict[str, Any],
    search_space: dict[str, Any],
) -> dict[str, Any]:
    """Cast string param values from MLflow back to the types Optuna expects."""
    out: dict[str, Any] = {}
    for name, val in raw.items():
        if name not in search_space:
            continue  # unknown param; skip silently
        spec = search_space[name]
        t = spec["type"]
        try:
            if t == "int":
                out[name] = int(float(str(val)))
            elif t == "float":
                out[name] = float(str(val))
            else:  # categorical — keep as string
                out[name] = str(val)
        except (TypeError, ValueError):
            out[name] = val  # leave as-is; Optuna will validate
    return out


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
            "--mlflow_parent_run_name", f"trial={trial.number}__asset={asset}__parent",
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
            "run_role": "sweep_summary",
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
    parser.add_argument(
        "--study_storage",
        default=None,
        help=(
            "Optuna storage URL for persisting studies (e.g. sqlite:///optuna_studies.db). "
            "Required for --resume."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume existing studies: uses stable study names (no timestamp) and "
            "load_if_exists=True. Requires --study_storage."
        ),
    )
    parser.add_argument(
        "--enqueue_best_from",
        default=None,
        metavar="EXPERIMENT",
        help=(
            "MLflow experiment name to warm-start from. For each asset, the best "
            "structural params logged by a previous sweep in that experiment are "
            "enqueued as the first trial of the new study."
        ),
    )
    args = parser.parse_args()

    if args.resume and not args.study_storage:
        parser.error("--resume requires --study_storage (e.g. sqlite:///optuna_studies.db)")

    cfg = load_sweep_config(args.config)
    sweep_cfg = cfg["sweep"]
    search_space = cfg["search_space"]

    n_trials = args.n_trials if args.n_trials is not None else int(sweep_cfg["n_trials"])
    assets = args.assets if args.assets is not None else list(sweep_cfg["assets"])
    experiment_name = str(sweep_cfg["mlflow_experiment"])
    # base_args from config (e.g. --horizon, --min_high_vol_recall_delta); always include recall floor
    cfg_base_args: list[str] = [str(a) for a in sweep_cfg.get("base_args", [])]
    base_args = cfg_base_args if "--min_high_vol_recall_delta" in cfg_base_args else cfg_base_args + ["--min_high_vol_recall_delta", "-1.0"]

    mlflow.set_tracking_uri("file:./mlruns")

    for asset in assets:
        asset_safe = asset.replace("^", "").replace("-", "_")

        # Stable name for resume; timestamped name for fresh runs.
        if args.resume:
            study_name = f"sweep_{asset_safe}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"sweep_{asset_safe}_{timestamp}"

        study_kwargs: dict[str, Any] = dict(
            study_name=study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        if args.study_storage:
            study_kwargs["storage"] = args.study_storage
            study_kwargs["load_if_exists"] = args.resume

        study = optuna.create_study(**study_kwargs)
        already_done = len(study.trials)

        if args.resume and already_done > 0:
            print(
                f"\n[sweep] Resuming study={study_name} | asset={asset} | "
                f"trials already done={already_done} | target={n_trials}"
            )
        else:
            print(
                f"\n[sweep] Starting Optuna study for asset={asset} | "
                f"study={study_name} | n_trials={n_trials}"
            )

        # Warm-start: enqueue best structural params from a previous experiment.
        # Only inject when the study is fresh (no existing trials) to avoid duplicates.
        if args.enqueue_best_from and already_done == 0:
            raw_params = get_best_params_from_experiment(args.enqueue_best_from, asset)
            if raw_params:
                coerced = _coerce_enqueued_params(raw_params, search_space)
                print(
                    f"[sweep] Warm-start enqueue for asset={asset} "
                    f"from experiment '{args.enqueue_best_from}': {coerced}"
                )
                study.enqueue_trial(coerced)
            else:
                print(
                    f"[sweep] No warm-start params found for asset={asset} "
                    f"in experiment '{args.enqueue_best_from}'; starting from scratch."
                )

        # How many trials to run this session.
        remaining = max(0, n_trials - already_done)
        if remaining == 0:
            print(f"[sweep] asset={asset} | study already has {already_done} trials — nothing to do.")
            log_sweep_summary(experiment_name, asset, study, n_failed=0)
            continue

        n_failed = 0
        objective = make_objective(
            asset=asset,
            search_space=search_space,
            base_args=base_args,
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

        study.optimize(objective_with_count, n_trials=remaining)

        n_completed = sum(1 for t in study.trials if t.value is not None and math.isfinite(t.value))
        if n_completed > 0:
            best = study.best_params
            print(f"\n[sweep] asset={asset} | best_robust_score={study.best_value:.4f} | best_params={best}")
        else:
            print(f"\n[sweep] asset={asset} | no successful trials")

        log_sweep_summary(experiment_name, asset, study, n_failed)

    print("\n[sweep] Done.")


if __name__ == "__main__":
    main()
