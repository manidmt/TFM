# MLflow v1 Phase 1 — Design Spec

**Date**: 2026-03-14
**Author**: Manuel Díaz-Meco Terrés (supervised by Claude Code)
**Branch**: feature/mlflow
**Reference plan**: `docs/mlflow_v1_plan.md`

---

## Goal

Add reproducible experiment tracking to `walk_forward_chain_tab.py` via MLflow, without breaking the existing workflow. Filesystem outputs remain canonical; MLflow logs them additionally.

---

## Scope

**In scope (Phase 1)**:
- New tracking module `src/quant_risk/tracking/mlflow_utils.py`
- 6 new CLI flags in `walk_forward_chain_tab.py`
- Parent/child run hierarchy (one parent per invocation, one child per asset group)
- Log params, metrics, artifacts, tags as defined in `docs/mlflow_v1_plan.md`
- GKG dataset provenance from `gkg_ingest_runs` DuckDB table
- Unit tests for the tracking module
- `mlflow ^2.13` added to `pyproject.toml`

**Out of scope (Phase 2 and beyond)**:
- Logging in `build_features.py` or `ingest_gkg.py`
- Per-fold or per-config child runs
- Remote MLflow server setup
- Replacing CSV/JSON outputs with MLflow-only artifacts

---

## Module: `src/quant_risk/tracking/`

### Files

```
src/quant_risk/tracking/
├── __init__.py          # empty or re-exports MlflowConfig, is_enabled
└── mlflow_utils.py      # all tracking logic
```

### `MlflowConfig` dataclass

```python
@dataclass
class MlflowConfig:
    enabled: bool = False
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "walk_forward_chain_tab"
    parent_run_name: str = ""          # auto-generated if empty
    strict: bool = False               # abort on tracking error if True
    extra_tags: dict[str, str] = field(default_factory=dict)
```

### Public helper surface

| Function | Purpose |
|---|---|
| `is_enabled(cfg)` | Returns `cfg.enabled` |
| `get_git_sha()` | Returns short git SHA or `"unknown"` |
| `start_parent_run(cfg, tags)` | Sets experiment, starts parent MLflow run, returns run or None |
| `start_child_run(cfg, run_name, tags)` | Starts nested child run, returns run or None |
| `log_params_safe(cfg, params)` | Logs dict of params; swallows errors unless strict |
| `log_metrics_safe(cfg, metrics)` | Logs dict of metrics; swallows errors unless strict |
| `log_artifact_safe(cfg, path)` | Logs a file artifact; swallows errors unless strict |
| `log_dict_artifact(cfg, data, filename)` | Serialises dict to temp JSON and logs it |
| `read_latest_gkg_manifest(db_path)` | Queries `gkg_ingest_runs` for the latest row; returns dict or None |
| `build_run_context(args, configs, git_sha)` | Builds a JSON-serialisable run-context dict |

All safe helpers follow this pattern:
```python
try:
    mlflow.<operation>(...)
except Exception as exc:
    if cfg.strict:
        raise
    warnings.warn(f"[mlflow] {exc}")
```

---

## CLI additions to `walk_forward_chain_tab.py`

| Flag | Default | Purpose |
|---|---|---|
| `--use_mlflow` | `False` (store_true) | Enable MLflow tracking |
| `--mlflow_tracking_uri` | `"file:./mlruns"` | Tracking store URI |
| `--mlflow_experiment` | `"walk_forward_chain_tab"` | Experiment name |
| `--mlflow_parent_run_name` | `""` | Override parent run name (auto if empty) |
| `--mlflow_strict` | `False` (store_true) | Abort run on tracking failure |
| `--mlflow_tags_json` | `""` | Extra tags as JSON string `'{"key":"val"}'` |

---

## Run hierarchy

```
Experiment: walk_forward_chain_tab
└── Parent run: wf_chain_tab__h{horizon}__{tabular_model}__{YYYYMMDD_HHMMSS}
    ├── Tags: script, component, news_profile, feature_profile, git_sha, horizon, tabular_model
    ├── Artifacts: cli_args.json, datasources.yaml, features.yaml, run_context.json,
    │             best_by_asset.csv, final_vs_persistence.csv (if multi-asset)
    ├── Metrics: n_asset_groups, n_successful_children, mean_delta_test_macro_f1 (if available)
    │
    └── Child run: asset={asset_name}
        ├── Tags: asset, tickers, horizon, tabular_model, use_gkg_change_detector, ...
        ├── Params: ~40 params as listed in mlflow_v1_plan.md
        ├── Metrics: robust_score, n_folds_ok, mean_delta_macro_vs_persistence,
        │           delta_test_macro_f1_vs_persistence, ... (full list from plan)
        └── Artifacts: fold_metrics.csv, summary.csv, best.json, final_vs_persistence.csv,
                       dataset_manifest_gkg.json (if news enabled)
```

---

## Integration points in `walk_forward_chain_tab.py`

The script is modified minimally. Tracking calls are grouped into clearly marked sections:

1. **After `args = parser.parse_args()`**: build `MlflowConfig` from args, detect git SHA
2. **Before the asset loop**: start parent run, log parent params/tags/config artifacts
3. **At top of asset loop**: start child run
4. **After `best` is selected**: log child params + walk-forward metrics
5. **After `final_cmp` is computed**: log test metrics + per-asset artifacts
6. **After the asset loop exits**: log parent-level aggregates + `best_by_asset.csv`
7. **In `finally` block**: end parent run

No existing code paths are modified — tracking calls are purely additive.

---

## Dataset Provenance

When `use_gkg_change_detector` or any news feature is active, `read_latest_gkg_manifest()` queries:

```sql
SELECT * FROM gkg_ingest_runs ORDER BY run_ts DESC LIMIT 1
```

The result is logged as:
- Selected fields as MLflow params (`gkg_manifest_*` prefix)
- Full row as `dataset_manifest_gkg.json` artifact in the child run

If the table does not exist or the query fails, a warning is printed and the run continues.

---

## Failure Policy

| Condition | Behaviour |
|---|---|
| `--use_mlflow` not passed | No MLflow code executes at all |
| MLflow not installed | Import error caught; warning printed; run continues |
| Any logging call fails | Warning printed; run continues (unless `--mlflow_strict`) |
| `--mlflow_strict` set | Any tracking error raises and aborts the run |

---

## Tests: `tests/test_mlflow_utils.py`

| Test | What it verifies |
|---|---|
| `test_mlflow_config_defaults` | Default field values |
| `test_is_enabled_false_by_default` | `is_enabled` returns False when `enabled=False` |
| `test_log_params_safe_swallows_errors` | Passes a broken MLflow mock; no exception raised when strict=False |
| `test_log_params_safe_raises_in_strict_mode` | Same broken mock; raises when strict=True |
| `test_log_metrics_safe_swallows_errors` | Same pattern for metrics |
| `test_log_artifact_safe_swallows_missing_file` | Non-existent file path; no exception when strict=False |
| `test_read_latest_gkg_manifest_returns_none_no_table` | In-memory DuckDB with no table; returns None |
| `test_read_latest_gkg_manifest_returns_row` | In-memory DuckDB with one row; returns correct dict |
| `test_build_run_context_keys` | Output contains expected top-level keys |
| `test_get_git_sha_returns_string` | Returns a non-empty string (or "unknown") |

---

## Acceptance Criteria

- `walk_forward_chain_tab.py` runs normally with `--use_mlflow` absent
- With `--use_mlflow`, a parent run and child runs appear in `mlruns/`
- Selected metrics visible in `mlflow ui`
- `summary.csv`, `fold_metrics.csv`, `best.json`, `final_vs_persistence.csv` attached as artifacts
- GKG manifest logged when `gkg_ingest_runs` exists
- All existing tests continue to pass
- New `tests/test_mlflow_utils.py` passes

---

## Non-Goals

- Do not log one run per fold
- Do not replace CSV/JSON outputs
- Do not add remote server configuration
- Do not change selection logic
