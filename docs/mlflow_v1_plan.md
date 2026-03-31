# MLflow v1 Integration Plan

## Purpose

This document defines the `MLflow` integration we want for the first tracked-experiment
version of the repository.

The goal is not to redesign the research workflow. The goal is to:

- keep the existing walk-forward outputs and selection logic intact
- add reproducible experiment tracking on top
- bind tracked runs to the now-frozen `GKG v1` data contract
- make cross-run comparison reliable enough to choose the final production recipe

## Preconditions

Before `MLflow`, the data contract must already be stable.

That condition is now met through:

- [gkg_v1.md](/home/manidmt/TFM/quant-risk-tfm/docs/gkg_v1.md)
- [datasources.yaml](/home/manidmt/TFM/quant-risk-tfm/config/datasources.yaml)
- [features.yaml](/home/manidmt/TFM/quant-risk-tfm/config/features.yaml)
- `gkg_ingest_runs` manifest in DuckDB

`MLflow` should therefore track model research on top of a stable data baseline, not
compensate for an unstable one.

## Scope of MLflow v1

### In scope

- integrate tracking into [`walk_forward_chain_tab.py`](/home/manidmt/TFM/quant-risk-tfm/scripts/walk_forward_chain_tab.py)
- log run parameters, selected configuration, summary metrics, and existing artifacts
- log the active `GKG v1`/feature profile used by the run
- log the latest `gkg_ingest_runs` manifest row as dataset provenance when news is enabled
- keep filesystem outputs as they are today

### Explicitly out of scope for v1

- replacing CSV/JSON outputs with `MLflow` artifacts only
- logging one `MLflow` run per fold or per candidate configuration
- real-time tracking inside `build_features.py` and `ingest_gkg.py`
- remote production deployment concerns
- hyperparameter search orchestration

Those can be added later if they help, but they are not required for a clean first version.

## Design Principles

### 1. Existing filesystem outputs remain canonical

The current runner already produces the right local artifacts:

- `fold_metrics.csv`
- `summary.csv`
- `best.json`
- `final_vs_persistence.csv`
- `best_by_asset.csv`
- root-level `final_vs_persistence.csv` for multi-asset runs

`MLflow` should log those files as artifacts, not replace them.

This keeps local debugging and backward compatibility intact.

### 2. Tracking must be low-friction and opt-in/opt-out safe

The runner should still work without `MLflow`.

Recommended behavior:

- `MLflow` enabled by default only if explicitly requested
- if tracking fails and `strict=false`, the experiment still runs and local files are written
- if tracking fails and `strict=true`, the run should abort

### 3. Run hierarchy must stay small

We should not create a run explosion.

For `MLflow v1`, the right hierarchy is:

- one **parent run** per invocation of `walk_forward_chain_tab.py`
- one **child run** per `asset` group inside that invocation

We should **not** create:

- one run per fold
- one run per structural config
- one run per XGB/TabPFN config

Those details already live in `fold_metrics.csv` and `summary.csv`. Logging them as nested runs
would add complexity without enough value.

## Recommended Run Structure

### Parent run

Represents one invocation of the runner.

Suggested experiment name:

- `walk_forward_chain_tab`

Suggested parent run name pattern:

- `wf_chain_tab__h{horizon}__{tabular_model}__{timestamp}`

Suggested parent tags:

- `script=walk_forward_chain_tab.py`
- `component=walk_forward`
- `news_profile=gkg_v1_light`
- `feature_profile=gkg_v1_features_compact`
- `git_sha=<commit>`

### Child runs

One child per asset-group written to:

- `h{horizon}/asset_{asset_name}`

Suggested child run name pattern:

- `asset={asset_name}`

Suggested child tags:

- `asset=<asset_name>`
- `tickers=<joined_tickers>`
- `horizon=<horizon>`
- `tabular_model=<xgb|tabpfn>`
- `use_gkg_change_detector=<bool>`
- `use_gkg_change_gate=<bool>`
- `use_gkg_change_alpha=<bool>`

## What To Log

### Params

Log only stable scalar/string parameters that define the run.

#### Runner / model params

- `horizon`
- `tabular_model`
- `per_ticker`
- `grid_profile`
- `min_train_end`
- `max_valid_end`
- `valid_months`
- `step_months`
- `max_struct_configs`
- `max_model_configs`
- `min_folds_ok`
- `min_positive_rate`
- `min_non_transition_delta`
- `min_high_vol_recall_delta`
- `stability_lambda`
- `positive_rate_lambda`
- `non_transition_penalty_lambda`
- `transition_bonus_lambda`
- `prune_after_folds`
- `prune_delta_threshold`
- `use_blend`
- `blend_alphas`
- `blend_conf_betas`
- `class_threshold_grid`
- `gate_thresholds`
- `calibration_method`

#### GKG sidecar params

- `use_gkg_change_detector`
- `use_gkg_change_gate`
- `use_gkg_change_alpha`
- `gkg_change_model`
- `gkg_change_context_cols`
- `gkg_change_calibration_method`
- `gkg_change_gate_thresholds`
- `gkg_change_alpha_weights`
- `gkg_change_alpha_weights_by_asset`
- `gkg_change_blend_hook_weight`

#### Data / feature profile params

- `gkg_profile_name`
- `gkg_profile_version`
- `gkg_profile_mode`
- `gkg_start`
- `gkg_store_raw`
- `gkg_sample_interval_minutes`
- `gkg_max_files_per_day`
- `gkg_publication_lag_bdays`
- `news_feature_profile_name`
- `news_feature_profile_version`
- `news_source_table`
- `news_windows`
- `news_compact_mode`
- `news_include_interactions`

### Metrics

#### Parent run metrics

Aggregate run-level metrics:

- number of asset groups
- number of successful child runs
- maybe aggregate mean `delta_test_macro_f1_vs_persistence` across child runs if available

#### Child run metrics

These should come from the selected row and final comparison:

- `robust_score`
- `n_folds_ok`
- `mean_delta_macro_vs_persistence`
- `mean_delta_transition_macro_f1_vs_persistence`
- `mean_delta_non_transition_macro_f1_vs_persistence`
- `mean_delta_high_vol_recall_vs_persistence`
- `oof_delta_macro_vs_persistence`
- `delta_test_macro_f1_vs_persistence`
- `delta_transition_macro_f1_vs_persistence_test`
- `delta_non_transition_macro_f1_vs_persistence_test`
- `delta_test_high_vol_recall_vs_persistence`
- `selected_struct_id`
- `selected_model_id`
- `selected_blend_alpha`
- `selected_blend_beta`
- `selected_gate_threshold`
- `selected_change_gate_threshold`
- `selected_change_alpha_weight`

If the GKG sidecar is active, also log:

- `gkg_change_macro_f1_test`
- `gkg_change_auc_test`
- `gkg_change_brier_test`
- `change_gating_rate_test`
- `mean_p_change_test`
- `mean_effective_alpha_test`

### Artifacts

For each child run, log:

- `fold_metrics.csv`
- `summary.csv`
- `best.json`
- `final_vs_persistence.csv`

For the parent run, log:

- a snapshot of CLI arguments
- a snapshot of `config/datasources.yaml`
- a snapshot of `config/features.yaml`
- optional `run_context.json`
- root-level `best_by_asset.csv`
- root-level `final_vs_persistence.csv` if produced

## Dataset Provenance

This is critical and should be explicit in `MLflow`.

When news features are enabled and `gkg_ingest_runs` exists, the runner should query the latest
manifest row and log it as:

- params/tags for the key fields
- one artifact JSON, for example `dataset_manifest_gkg.json`

Recommended manifest fields to capture:

- `profile_name`
- `profile_version`
- `profile_mode`
- `start_date`
- `end_date`
- `topic_ids`
- `topics_hash`
- `store_raw`
- `sample_interval_minutes`
- `max_files_per_day`
- `max_files_total`
- `publication_lag_bdays`
- `request_pause_seconds`
- `timeout_seconds`
- `max_retries`
- `raw_table`
- `daily_table`

This ties tracked model results to the actual GKG dataset contract used.

## Implementation Architecture

### New module

Create:

- `src/quant_risk/tracking/mlflow_utils.py`

Responsibilities:

- lazy import of `mlflow`
- configuration/dataclass for tracking
- safe logging helpers for params/metrics/artifacts/tags
- run-context snapshot helpers
- git sha detection
- dataset manifest extraction from DuckDB

### Suggested helper surface

- `MlflowConfig`
- `is_enabled(cfg) -> bool`
- `start_parent_run(...)`
- `start_child_run(...)`
- `log_params_safe(...)`
- `log_metrics_safe(...)`
- `log_artifact_safe(...)`
- `log_dict_artifact(...)`
- `read_latest_gkg_manifest(db_path) -> dict | None`
- `build_run_context(args, configs, selected_table_info, git_sha) -> dict`

## CLI Additions

Recommended new flags in `walk_forward_chain_tab.py`:

- `--use_mlflow`
- `--mlflow_tracking_uri`
- `--mlflow_experiment`
- `--mlflow_parent_run_name`
- `--mlflow_strict`
- `--mlflow_tags_json`

Recommended defaults:

- `use_mlflow = false`
- `mlflow_tracking_uri = file:./mlruns`
- `mlflow_experiment = walk_forward_chain_tab`
- `mlflow_strict = false`

## Failure Policy

Default policy:

- if `MLflow` logging fails, print a warning and continue
- local filesystem outputs remain intact

Optional strict policy:

- if `--mlflow_strict` is enabled, tracking failures abort the run

This keeps experimentation resilient while still allowing hard enforcement when needed.

## Implementation Order

### Phase 1

- add `mlflow` dependency
- add tracking utility module
- integrate parent/child run tracking into `walk_forward_chain_tab.py`
- log configs, selected metrics, and artifacts

### Phase 2

- optionally add lightweight logging to `scripts/build_features.py`
- optionally add lightweight logging to `scripts/ingest_gkg.py`

These second-phase steps are useful, but the main research bottleneck is the walk-forward runner,
so that should land first.

## Acceptance Criteria

We should consider `MLflow v1` complete when:

- `walk_forward_chain_tab.py` runs normally with `MLflow` disabled
- with `MLflow` enabled, a parent run and child runs are created correctly
- selected metrics are visible in the UI
- `summary.csv`, `fold_metrics.csv`, `best.json`, and `final_vs_persistence.csv` are attached
- GKG dataset provenance is logged from `gkg_ingest_runs`
- local filesystem outputs stay unchanged

## Why This Design

This approach is intentionally conservative:

- minimal invasive changes to the runner
- no duplication of selection logic
- no run explosion
- no new ambiguity about dataset version

It gives us serious experiment tracking quickly, without destabilizing the workflow that already
works.
