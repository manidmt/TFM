# Experiment Sweep Runner — Design Spec

**Date**: 2026-03-15
**Branch**: feature/mlflow
**Scope**: Optuna-based sweep runner over model combinations, logged to MLflow

---

## Goal

Enable systematic exploration of model combinations (chain variant, tabular model, GKG detector, key hyperparams) across individual assets, with results fully tracked in MLflow. Each asset gets an independent Optuna study (60 trials, sequential, TPE sampler) optimising `robust_score`.

---

## Context

- `walk_forward_chain_tab.py` already runs walk-forward evaluation and logs results to MLflow via `--use_mlflow`.
- Chain variant YAMLs live in `src/quant_risk/models/econometric/`: `chain_variants.yaml` (GARCH), `chain_variants_egarch.yaml`, `chain_variants_gjrgarch.yaml`, `chain_variants_har.yaml`. The existing CLI flag is `--chain_variants_config` (not `--chain_variants_file`).
- TabPFN hyperparams for the main model are currently defined in `src/quant_risk/models/tabular/tabpfn_variants.yaml` (via `--tabpfn_variants_config`). The sweep needs direct CLI flags to override individual hyperparams.
- Feature engineering params (`windows`, `compact_mode`, `include_interactions`) are consumed by `build_features.py` and stored in DuckDB — **out of scope** for this sweep.
- MLflow parent/child run hierarchy is already established. Extra tags passed via `--mlflow_tags_json` currently land only on the **parent run** (via `start_parent_run`). To enable per-child querying by trial, `start_child_run` in `mlflow_utils.py` must also merge `cfg.extra_tags` — this is one additional change to the tracking module.
- Existing relevant CLI flags (no addition needed): `--tabular_model`, `--asset`, `--use_blend` (store_true), `--use_gkg_change_detector` (store_true), `--gkg_change_model`, `--chain_variants_config`.

---

## Scope

**In scope**:
- New `config/sweep_config.yaml` defining search space, assets, and trial budget
- New `scripts/sweep_runner.py` running an Optuna study per asset
- New CLI flags in `walk_forward_chain_tab.py` for hyperparams not yet individually overridable
- One-line fix to `start_child_run` in `mlflow_utils.py` to propagate `extra_tags` to child runs
- Sweep summary MLflow run per asset logging best params and best score

**Out of scope**:
- Feature engineering hyperparams (windows, compact_mode, include_interactions)
- Parallel trial execution
- Multi-objective optimisation
- Remote MLflow server setup
- Modifying selection logic in `walk_forward_chain_tab.py`

---

## Files Changed

| File | Action |
|---|---|
| `config/sweep_config.yaml` | Create |
| `scripts/sweep_runner.py` | Create |
| `scripts/walk_forward_chain_tab.py` | Add ~5 new CLI flags |
| `src/quant_risk/tracking/mlflow_utils.py` | One-line fix to `start_child_run` |

---

## New CLI Flags for `walk_forward_chain_tab.py`

All new flags default to `None`. When `None`, the existing value from the variant config/YAML is used unchanged. This guarantees full backward compatibility.

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--tabpfn_n_estimators` | int | `None` | Override TabPFN n_estimators for the main model |
| `--tabpfn_softmax_temperature` | float | `None` | Override TabPFN softmax_temperature for the main model |
| `--xgb_max_depth` | int | `None` | Override XGB max_depth for the main model grid |
| `--xgb_learning_rate` | float | `None` | Override XGB learning_rate for the main model grid |
| `--gkg_tabpfn_n_estimators` | int | `None` | Override TabPFN n_estimators for the GKG change detector |

**Note on `--use_blend`**: This flag already exists as `store_true`. In the sweep runner, when `use_blend="true"` is sampled, `--use_blend` is appended to CLI args. When `"false"`, it is omitted.

**Note on `--use_gkg_change_detector`**: This flag already exists as `store_true`. Identical rule: when `use_gkg_change_detector="true"` is sampled, `--use_gkg_change_detector` is appended to CLI args. When `"false"`, it is omitted. Never pass `--use_gkg_change_detector true` — it is a valueless flag.

**Note on `xgb_n_estimators`**: Not added to the search space. The chain variants YAML already defines `n_estimators` grids per structural config; holding it at grid defaults avoids conflating grid-level search with sweep-level search.

**Note on `--gkg_tabpfn_n_estimators` consumption path**: This flag requires three internal wiring changes to `walk_forward_chain_tab.py` beyond the argparse addition:
1. Add `tabpfn_n_estimators: int | None = None` parameter to `_fit_eval_gkg_change_detector`.
2. Inside `_fit_eval_gkg_change_detector`, pass it to `GkgChangeDetectorConfig` conditionally: `GkgChangeDetectorConfig(..., **({"tabpfn_n_estimators": tabpfn_n_estimators} if tabpfn_n_estimators is not None else {}))`.
3. Thread `args.gkg_tabpfn_n_estimators` through both call sites of `_fit_eval_gkg_change_detector` (currently at lines ~2053 and ~2430).
The `GkgChangeDetectorConfig` dataclass already has the `tabpfn_n_estimators` field (added in this branch).

---

## Fix to `mlflow_utils.py`

`start_child_run` currently does not merge `cfg.extra_tags`. Extra tags passed via `--mlflow_tags_json` (including `sweep_trial` and `optuna_study`) only land on the parent run. To make per-trial MLflow queries reliable, `start_child_run` must also merge `cfg.extra_tags` into the child run tags:

The fix goes inside the existing `try` block (matching the actual function structure):

```python
# Before (current — inside try block)
return mlflow.start_run(run_name=run_name, nested=True, tags=tags or {})

# After
try:
    merged = {**(tags or {}), **cfg.extra_tags}
    return mlflow.start_run(run_name=run_name, nested=True, tags=merged)
except Exception as exc:
    _warn_or_raise(cfg, exc)
    return None
```

---

## `config/sweep_config.yaml`

```yaml
sweep:
  n_trials: 60
  mlflow_experiment: "walk_fordward_sweep_adv"
  assets:
    - "^GSPC"
    - "BTC-USD"
    - "TLT"

search_space:
  # Discrete structural choices
  chain_variants_config:
    type: categorical
    choices:
      - src/quant_risk/models/econometric/chain_variants.yaml
      - src/quant_risk/models/econometric/chain_variants_egarch.yaml
      - src/quant_risk/models/econometric/chain_variants_gjrgarch.yaml
      - src/quant_risk/models/econometric/chain_variants_har.yaml

  tabular_model:
    type: categorical
    choices: [xgb, tabpfn]

  use_gkg_change_detector:
    type: categorical
    choices: ["true", "false"]

  gkg_change_model:
    type: categorical
    choices: [logit, xgb, tabpfn]

  # TabPFN hyperparams — active when tabular_model=tabpfn
  tabpfn_n_estimators:
    type: int
    low: 4
    high: 16

  tabpfn_softmax_temperature:
    type: float
    low: 0.5
    high: 1.5
    # Note: existing calibrated values are in [0.8, 0.9]; range is intentionally wider for exploration

  # XGB hyperparams — active when tabular_model=xgb
  xgb_max_depth:
    type: int
    low: 3
    high: 8

  xgb_learning_rate:
    type: float
    low: 0.01
    high: 0.2
    log: true

  # GKG TabPFN hyperparams — active when gkg_change_model=tabpfn AND use_gkg_change_detector=true
  gkg_tabpfn_n_estimators:
    type: int
    low: 4
    high: 16

  # Blending — use_blend is store_true; sampled as "true"/"false" string
  use_blend:
    type: categorical
    choices: ["true", "false"]
```

**Conditional sampling rules** (applied in `sweep_runner.py`):
- `tabpfn_n_estimators`, `tabpfn_softmax_temperature`: always sampled; only passed to CLI when `tabular_model=tabpfn`
- `xgb_max_depth`, `xgb_learning_rate`: always sampled; only passed to CLI when `tabular_model=xgb`
- `gkg_change_model`: always sampled; only passed to CLI when `use_gkg_change_detector=true`
- `gkg_tabpfn_n_estimators`: always sampled; only passed to CLI when `use_gkg_change_detector=true` AND `gkg_change_model=tabpfn`
- `use_blend`: when `"true"`, append `--use_blend` (no value); when `"false"`, omit

---

## `scripts/sweep_runner.py`

**Key functions**:

**`study_name` must be asset-scoped**: each per-asset Optuna study is created with a name of the form `f"sweep_{asset}_{YYYYMMDD_HHMMSS}"`. This ensures the triple-filter `(asset, sweep_trial, optuna_study)` in the MLflow query is unique even if multiple sweep processes run concurrently against the same experiment.

| Function | Purpose |
|---|---|
| `load_sweep_config(path)` | Load and validate YAML; return config dict |
| `build_trial_args(trial, search_space, asset, base_args)` | Sample Optuna params; return list of CLI args applying conditional rules |
| `run_trial_subprocess(cli_args, repo_root)` | Subprocess call; return exit_code |
| `read_robust_score_from_mlflow(experiment_name, asset, trial_number, study_name)` | Query MLflow for latest child run matching asset + sweep_trial + optuna_study; return float or None |
| `make_objective(asset, search_space, base_args, experiment_name, study_name)` | Returns Optuna objective closure for one asset; `study_name` must be asset-scoped (e.g., `f"sweep_{asset}_{timestamp}"`) to guarantee query uniqueness |
| `log_sweep_summary(experiment_name, asset, study)` | Log best params + best score + optuna_study_summary.json to a sweep summary run |
| `main()` | Entry point; parse args, loop over assets, run studies, log summaries |

**Subprocess call pattern**:
```python
cmd = [
    "poetry", "run", "python", "scripts/walk_forward_chain_tab.py",
    "--use_mlflow",
    "--mlflow_experiment", experiment_name,
    "--mlflow_tags_json", json.dumps({
        "sweep_trial": str(trial.number),
        "optuna_study": study.study_name,
    }),
    "--asset", asset,
    *trial_specific_args,
]
subprocess.run(cmd, cwd=repo_root, check=False)
```

**MLflow query pattern** (after subprocess returns):
```python
runs = mlflow.search_runs(
    experiment_names=[experiment_name],
    filter_string=(
        f"tags.asset = '{asset}' "
        f"AND tags.sweep_trial = '{trial.number}' "
        f"AND tags.optuna_study = '{study.study_name}'"
    ),
    order_by=["start_time DESC"],
    max_results=1,
)
if runs.empty:
    return float("-inf")
score = runs.iloc[0].get("metrics.robust_score", float("nan"))
if score is None or (isinstance(score, float) and (score != score)):  # NaN check
    return float("-inf")
return float(score)
```

---

## MLflow Run Hierarchy

```
Experiment: walk_fordward_sweep_adv
│
├── Per-trial runs (created by walk_forward_chain_tab.py)
│   └── Parent run: wf_chain_tab__h{horizon}__{model}__{timestamp}
│       ├── Tags: sweep_trial=<N>, optuna_study=<study_name>  ← from --mlflow_tags_json
│       └── Child run: asset={asset_name}
│           ├── Tags: asset, sweep_trial=<N>, optuna_study=<study_name>  ← propagated via extra_tags fix
│           └── Metrics: robust_score, delta_test_macro_f1_vs_persistence, ...
│
└── Sweep summary runs (created by sweep_runner.py — one per asset)
    └── sweep__{asset}__{YYYYMMDD_HHMMSS}
        ├── Tags: asset, optuna_study_name, n_trials
        ├── Params: best trial params (all sampled params)
        ├── Metrics: best_robust_score, n_successful_trials, n_failed_trials
        └── Artifact: optuna_study_summary.json (all trial params + scores)
```

The sweep runner only **reads** per-trial runs; it never writes into them. Sweep summary runs are purely additive.

---

## Failure Policy

| Condition | Behaviour |
|---|---|
| Trial subprocess fails (non-zero exit) | `robust_score = -inf`; warning logged with trial number and CLI args; study continues |
| MLflow query returns no matching run | `robust_score = -inf`; warning logged; study continues |
| MLflow run found but `robust_score` is NaN or missing | `robust_score = -inf`; warning logged; study continues |
| `--sweep_strict` flag set | Any failure raises and aborts the sweep |
| Asset not found in walk-forward output | Treated as subprocess failure |

---

## Acceptance Criteria

- `python scripts/sweep_runner.py --config config/sweep_config.yaml --n_trials 2` runs end-to-end for a single asset
- Each trial creates a parent + child run in the `walk_fordward_sweep_adv` experiment with `sweep_trial` and `optuna_study` tags on both parent and child
- After all trials, a sweep summary run exists with best params and `best_robust_score`
- `walk_forward_chain_tab.py` runs normally without any new flags (backward compatible)
- With `--chain_variants_config src/quant_risk/models/econometric/chain_variants_egarch.yaml`, the EGARCH variant is used
- All existing tests continue to pass

---

## Non-Goals

- Do not vary feature engineering params (windows, compact_mode, include_interactions)
- Do not implement parallel trials
- Do not implement multi-objective optimisation
- Do not add a remote MLflow server
- Do not modify selection logic in `walk_forward_chain_tab.py`
