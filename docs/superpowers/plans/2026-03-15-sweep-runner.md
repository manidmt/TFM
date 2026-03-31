# Experiment Sweep Runner — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Optuna-based sweep runner that explores model combinations per asset and logs everything to MLflow.

**Architecture:** Four independent changes land in sequence: (1) propagate MLflow extra_tags to child runs, (2) expose hyperparam overrides as CLI flags in walk_forward_chain_tab.py, (3) wire gkg_tabpfn_n_estimators through the GKG detector call stack, (4) create sweep_config.yaml + sweep_runner.py with unit-tested core functions.

**Tech Stack:** Python, Optuna (`optuna`), MLflow (`mlflow`), PyYAML, subprocess, pytest

---

## Chunk 1: MLflow fix + CLI flags + GKG wiring

### Task 1: Propagate extra_tags to child MLflow runs

**Files:**
- Modify: `src/quant_risk/tracking/mlflow_utils.py:94-95`
- Modify: `tests/test_mlflow_utils.py` (add one test)

**Spec ref:** `docs/superpowers/specs/2026-03-15-sweep-runner-design.md` — "Fix to mlflow_utils.py"

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mlflow_utils.py`:

```python
def test_start_child_run_propagates_extra_tags():
    cfg = MlflowConfig(enabled=True, extra_tags={"sweep_trial": "7", "optuna_study": "sweep_GSPC_20260315"})
    explicit_tags = {"asset": "^GSPC"}

    with patch("quant_risk.tracking.mlflow_utils.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value = MagicMock()
        start_child_run(cfg, run_name="asset=^GSPC", tags=explicit_tags)

    call_kwargs = mock_mlflow.start_run.call_args[1]
    merged = call_kwargs["tags"]
    assert merged["asset"] == "^GSPC"
    assert merged["sweep_trial"] == "7"
    assert merged["optuna_study"] == "sweep_GSPC_20260315"
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_mlflow_utils.py::test_start_child_run_propagates_extra_tags -v
```

Expected: FAILED — `sweep_trial` and `optuna_study` are absent from the merged tags.

- [ ] **Step 3: Apply the fix to `start_child_run`**

In `src/quant_risk/tracking/mlflow_utils.py`, find the `start_child_run` function. The current `try` block is:

```python
    try:
        return mlflow.start_run(run_name=run_name, nested=True, tags=tags or {})
    except Exception as exc:
        _warn_or_raise(cfg, exc)
        return None
```

Replace with:

```python
    try:
        merged = {**(tags or {}), **cfg.extra_tags}
        return mlflow.start_run(run_name=run_name, nested=True, tags=merged)
    except Exception as exc:
        _warn_or_raise(cfg, exc)
        return None
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_mlflow_utils.py -v
```

Expected: all tests including the new one PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && git add src/quant_risk/tracking/mlflow_utils.py tests/test_mlflow_utils.py && git commit -m "fix(mlflow): propagate extra_tags to child runs in start_child_run"
```

---

### Task 2: CLI flags for hyperparam overrides in walk_forward_chain_tab.py

**Files:**
- Modify: `scripts/walk_forward_chain_tab.py` (argparse section ~line 2815, config-build section ~line 3023)
- Modify: `tests/test_walk_forward_mlflow_smoke.py` (add one test)

**Spec ref:** "New CLI Flags for walk_forward_chain_tab.py"

- [ ] **Step 1: Write the failing test**

Append to `tests/test_walk_forward_mlflow_smoke.py`:

```python
def test_new_cli_flags_exist():
    """All 5 new CLI flags are registered and default to None."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "wf", "scripts/walk_forward_chain_tab.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wf"] = mod
    spec.loader.exec_module(mod)

    import argparse
    parser = argparse.ArgumentParser()
    # Re-use the script's parser by calling the setup function
    # The script exposes main() which calls _run(); we just parse with defaults
    args = mod._parse_args([])  # will raise AttributeError if function doesn't exist
    assert args.tabpfn_n_estimators is None
    assert args.tabpfn_softmax_temperature is None
    assert args.xgb_max_depth is None
    assert args.xgb_learning_rate is None
    assert args.gkg_tabpfn_n_estimators is None
```

> **Note**: `_parse_args` does not yet exist — this test will guide its creation. Alternatively, skip the function approach and test by running `--help` and checking the output. Use the simpler approach below if `_parse_args` is complex to add.

**Simpler version** (use this instead):

```python
def test_new_cli_flags_in_help():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "scripts/walk_forward_chain_tab.py", "--help"],
        capture_output=True, text=True,
        cwd="/home/manidmt/TFM/quant-risk-tfm"
    )
    help_text = result.stdout + result.stderr
    assert "--tabpfn_n_estimators" in help_text
    assert "--tabpfn_softmax_temperature" in help_text
    assert "--xgb_max_depth" in help_text
    assert "--xgb_learning_rate" in help_text
    assert "--gkg_tabpfn_n_estimators" in help_text
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_walk_forward_mlflow_smoke.py::test_new_cli_flags_in_help -v
```

Expected: FAILED — flags not yet in `--help`.

- [ ] **Step 3: Add the 5 new argparse flags**

In `scripts/walk_forward_chain_tab.py`, after the `--tabpfn_n_preprocessing_jobs` line (~line 2817), add:

```python
    parser.add_argument(
        "--tabpfn_n_estimators",
        type=int,
        default=None,
        help="Override TabPFN n_estimators for the main model (None = use variant config).",
    )
    parser.add_argument(
        "--tabpfn_softmax_temperature",
        type=float,
        default=None,
        help="Override TabPFN softmax_temperature for the main model (None = use variant config).",
    )
    parser.add_argument(
        "--xgb_max_depth",
        type=int,
        default=None,
        help="Override XGB max_depth for all model grid configs (None = use grid defaults).",
    )
    parser.add_argument(
        "--xgb_learning_rate",
        type=float,
        default=None,
        help="Override XGB learning_rate for all model grid configs (None = use grid defaults).",
    )
    parser.add_argument(
        "--gkg_tabpfn_n_estimators",
        type=int,
        default=None,
        help="Override TabPFN n_estimators for the GKG change detector (None = use config default).",
    )
```

- [ ] **Step 4: Add override logic after `xgb_cfgs` is built**

In `scripts/walk_forward_chain_tab.py`, find the block where `xgb_cfgs` is built (~lines 3023–3039):

```python
    if str(args.tabular_model).lower() == "tabpfn":
        xgb_cfgs = make_tabpfn_configs(
            ...
        )
    else:
        xgb_cfgs = make_xgb_configs(args.grid_profile)
    ...
    if cfg_cap > 0:
        xgb_cfgs = xgb_cfgs[:cfg_cap]
```

After the `xgb_cfgs = xgb_cfgs[:cfg_cap]` line, add:

```python
    # Apply per-flag hyperparam overrides (None = keep grid defaults)
    if str(args.tabular_model).lower() == "tabpfn":
        if args.tabpfn_n_estimators is not None:
            xgb_cfgs = [{**c, "n_estimators": int(args.tabpfn_n_estimators)} for c in xgb_cfgs]
        if args.tabpfn_softmax_temperature is not None:
            xgb_cfgs = [{**c, "softmax_temperature": float(args.tabpfn_softmax_temperature)} for c in xgb_cfgs]
    else:
        if args.xgb_max_depth is not None:
            xgb_cfgs = [{**c, "max_depth": int(args.xgb_max_depth)} for c in xgb_cfgs]
        if args.xgb_learning_rate is not None:
            xgb_cfgs = [{**c, "learning_rate": float(args.xgb_learning_rate)} for c in xgb_cfgs]
```

- [ ] **Step 5: Run to verify test passes**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_walk_forward_mlflow_smoke.py::test_new_cli_flags_in_help -v
```

Expected: PASSED.

- [ ] **Step 6: Run the full test suite**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && git add scripts/walk_forward_chain_tab.py tests/test_walk_forward_mlflow_smoke.py && git commit -m "feat(sweep): add hyperparam override CLI flags to walk_forward_chain_tab"
```

---

### Task 3: Wire gkg_tabpfn_n_estimators through _fit_eval_gkg_change_detector

**Files:**
- Modify: `scripts/walk_forward_chain_tab.py:1082-1097` (function signature)
- Modify: `scripts/walk_forward_chain_tab.py:1150-1156` (GkgChangeDetectorConfig construction)
- Modify: `scripts/walk_forward_chain_tab.py:2053-2067` (call site 1)
- Modify: `scripts/walk_forward_chain_tab.py:2430-2444` (call site 2)
- Modify: `tests/test_walk_forward_mlflow_smoke.py` (add one test)

**Spec ref:** "Note on --gkg_tabpfn_n_estimators consumption path"

- [ ] **Step 1: Write the failing test**

Append to `tests/test_walk_forward_mlflow_smoke.py`:

```python
def test_gkg_tabpfn_n_estimators_wired():
    """_fit_eval_gkg_change_detector accepts tabpfn_n_estimators kwarg."""
    import inspect, importlib.util, sys
    spec_mod = importlib.util.spec_from_file_location("wf2", "scripts/walk_forward_chain_tab.py")
    mod = importlib.util.module_from_spec(spec_mod)
    sys.modules["wf2"] = mod
    spec_mod.loader.exec_module(mod)

    sig = inspect.signature(mod._fit_eval_gkg_change_detector)
    assert "tabpfn_n_estimators" in sig.parameters
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_walk_forward_mlflow_smoke.py::test_gkg_tabpfn_n_estimators_wired -v
```

Expected: FAILED — parameter not yet present.

- [ ] **Step 3: Add `tabpfn_n_estimators` parameter to `_fit_eval_gkg_change_detector`**

In `scripts/walk_forward_chain_tab.py`, find the `_fit_eval_gkg_change_detector` function signature (lines 1082–1097). Add the new parameter after `xgb_n_jobs`:

```python
def _fit_eval_gkg_change_detector(
    *,
    feature_cols: Sequence[str],
    x_train_aligned: np.ndarray,
    y_train_target_aligned: np.ndarray,
    regime_prev_train: np.ndarray,
    persist_train_pred: np.ndarray,
    x_eval_aligned: np.ndarray,
    y_eval_target_aligned: np.ndarray,
    regime_prev_eval: np.ndarray,
    persist_eval_pred: np.ndarray,
    model_type: str,
    calibration_method: str,
    context_cols: tuple[str, ...],
    xgb_n_jobs: int,
    tabpfn_n_estimators: int | None = None,
) -> dict[str, Any]:
```

- [ ] **Step 4: Update `GkgChangeDetectorConfig` construction inside the function**

Find the `GkgChangeDetectorConfig(...)` call at lines 1150–1156:

```python
    cfg = GkgChangeDetectorConfig(
        model_type=str(model_type).lower(),
        calibration_method=str(calibration_method).lower(),
        random_state=42,
        seed=42,
        xgb_n_jobs=int(xgb_n_jobs),
    )
```

Replace with:

```python
    _tabpfn_override = (
        {"tabpfn_n_estimators": int(tabpfn_n_estimators)}
        if tabpfn_n_estimators is not None
        else {}
    )
    cfg = GkgChangeDetectorConfig(
        model_type=str(model_type).lower(),
        calibration_method=str(calibration_method).lower(),
        random_state=42,
        seed=42,
        xgb_n_jobs=int(xgb_n_jobs),
        **_tabpfn_override,
    )
```

- [ ] **Step 5: Thread `args.gkg_tabpfn_n_estimators` through both call sites**

**Call site 1** (~line 2053): Find `_fit_eval_gkg_change_detector(` with `xgb_n_jobs=int(xgb_n_jobs),` as the last arg. Add:

```python
            xgb_n_jobs=int(xgb_n_jobs),
            tabpfn_n_estimators=args.gkg_tabpfn_n_estimators,
```

**Call site 2** (~line 2430): Same pattern — add `tabpfn_n_estimators=args.gkg_tabpfn_n_estimators,` after `xgb_n_jobs=int(xgb_n_jobs),`.

- [ ] **Step 6: Run tests**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_walk_forward_mlflow_smoke.py::test_gkg_tabpfn_n_estimators_wired -v && poetry run pytest --tb=short -q
```

Expected: new test PASSES, full suite still green.

- [ ] **Step 7: Commit**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && git add scripts/walk_forward_chain_tab.py tests/test_walk_forward_mlflow_smoke.py && git commit -m "feat(sweep): wire gkg_tabpfn_n_estimators through _fit_eval_gkg_change_detector"
```

---

## Chunk 2: Sweep config + runner

### Task 4: Create sweep_config.yaml

**Files:**
- Create: `config/sweep_config.yaml`

No tests — this is a config file; correctness is validated by `load_sweep_config` in Task 5.

- [ ] **Step 1: Create the file**

Create `config/sweep_config.yaml` with exactly this content:

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

  # TabPFN hyperparams — passed to CLI only when tabular_model=tabpfn
  tabpfn_n_estimators:
    type: int
    low: 4
    high: 16

  tabpfn_softmax_temperature:
    type: float
    low: 0.5
    high: 1.5

  # XGB hyperparams — passed to CLI only when tabular_model=xgb
  xgb_max_depth:
    type: int
    low: 3
    high: 8

  xgb_learning_rate:
    type: float
    low: 0.01
    high: 0.2
    log: true

  # GKG TabPFN hyperparams — passed to CLI only when use_gkg_change_detector=true AND gkg_change_model=tabpfn
  gkg_tabpfn_n_estimators:
    type: int
    low: 4
    high: 16

  # Blending — store_true flag: "true" appends --use_blend, "false" omits it
  use_blend:
    type: categorical
    choices: ["true", "false"]
```

- [ ] **Step 2: Commit**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && git add config/sweep_config.yaml && git commit -m "feat(sweep): add sweep_config.yaml search space"
```

---

### Task 5: Create sweep_runner.py

**Files:**
- Create: `scripts/sweep_runner.py`
- Create: `tests/test_sweep_runner.py`

**Spec ref:** `docs/superpowers/specs/2026-03-15-sweep-runner-design.md` — "`scripts/sweep_runner.py`" section

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sweep_runner.py`:

```python
"""Unit tests for sweep_runner.py."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

# Add scripts/ to path so we can import sweep_runner
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import sweep_runner


# ---------------------------------------------------------------------------
# load_sweep_config
# ---------------------------------------------------------------------------

def test_load_sweep_config_returns_dict(tmp_path):
    cfg = {
        "sweep": {"n_trials": 5, "mlflow_experiment": "test_exp", "assets": ["^GSPC"]},
        "search_space": {
            "tabular_model": {"type": "categorical", "choices": ["xgb", "tabpfn"]},
        },
    }
    p = tmp_path / "sweep.yaml"
    p.write_text(yaml.dump(cfg))
    result = sweep_runner.load_sweep_config(str(p))
    assert result["sweep"]["n_trials"] == 5
    assert result["sweep"]["assets"] == ["^GSPC"]
    assert "search_space" in result


def test_load_sweep_config_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        sweep_runner.load_sweep_config("/nonexistent/path.yaml")


# ---------------------------------------------------------------------------
# build_trial_args
# ---------------------------------------------------------------------------

def _make_trial(params: dict):
    """Return a mock Optuna trial that returns params on suggest_* calls."""
    trial = MagicMock()
    trial.number = 3

    def suggest_categorical(name, choices):
        return params.get(name, choices[0])

    def suggest_int(name, low, high, **kw):
        return params.get(name, low)

    def suggest_float(name, low, high, **kw):
        return params.get(name, low)

    trial.suggest_categorical.side_effect = suggest_categorical
    trial.suggest_int.side_effect = suggest_int
    trial.suggest_float.side_effect = suggest_float
    return trial


def test_build_trial_args_tabpfn_passes_tabpfn_flags():
    search_space = {
        "tabular_model": {"type": "categorical", "choices": ["tabpfn"]},
        "tabpfn_n_estimators": {"type": "int", "low": 4, "high": 16},
        "tabpfn_softmax_temperature": {"type": "float", "low": 0.5, "high": 1.5},
        "xgb_max_depth": {"type": "int", "low": 3, "high": 8},
        "use_blend": {"type": "categorical", "choices": ["false"]},
        "use_gkg_change_detector": {"type": "categorical", "choices": ["false"]},
        "gkg_change_model": {"type": "categorical", "choices": ["logit"]},
        "gkg_tabpfn_n_estimators": {"type": "int", "low": 4, "high": 16},
        "chain_variants_config": {"type": "categorical", "choices": ["config/chain_variants.yaml"]},
    }
    params = {
        "tabular_model": "tabpfn",
        "tabpfn_n_estimators": 8,
        "tabpfn_softmax_temperature": 0.9,
        "use_blend": "false",
        "use_gkg_change_detector": "false",
        "gkg_change_model": "logit",
        "gkg_tabpfn_n_estimators": 6,
        "xgb_max_depth": 5,
        "chain_variants_config": "config/chain_variants.yaml",
    }
    trial = _make_trial(params)
    args = sweep_runner.build_trial_args(trial, search_space, "^GSPC", [])
    args_str = " ".join(args)
    assert "--tabular_model tabpfn" in args_str
    assert "--tabpfn_n_estimators 8" in args_str
    assert "--tabpfn_softmax_temperature" in args_str
    # XGB flags must NOT appear when tabular_model=tabpfn
    assert "--xgb_max_depth" not in args_str


def test_build_trial_args_xgb_passes_xgb_flags():
    search_space = {
        "tabular_model": {"type": "categorical", "choices": ["xgb"]},
        "tabpfn_n_estimators": {"type": "int", "low": 4, "high": 16},
        "tabpfn_softmax_temperature": {"type": "float", "low": 0.5, "high": 1.5},
        "xgb_max_depth": {"type": "int", "low": 3, "high": 8},
        "xgb_learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
        "use_blend": {"type": "categorical", "choices": ["true"]},
        "use_gkg_change_detector": {"type": "categorical", "choices": ["false"]},
        "gkg_change_model": {"type": "categorical", "choices": ["logit"]},
        "gkg_tabpfn_n_estimators": {"type": "int", "low": 4, "high": 16},
        "chain_variants_config": {"type": "categorical", "choices": ["config/chain_variants.yaml"]},
    }
    params = {
        "tabular_model": "xgb",
        "xgb_max_depth": 5,
        "xgb_learning_rate": 0.05,
        "use_blend": "true",
        "use_gkg_change_detector": "false",
        "gkg_change_model": "logit",
        "gkg_tabpfn_n_estimators": 4,
        "tabpfn_n_estimators": 8,
        "tabpfn_softmax_temperature": 0.9,
        "chain_variants_config": "config/chain_variants.yaml",
    }
    trial = _make_trial(params)
    args = sweep_runner.build_trial_args(trial, search_space, "^GSPC", [])
    args_str = " ".join(args)
    assert "--xgb_max_depth 5" in args_str
    assert "--xgb_learning_rate" in args_str
    assert "--use_blend" in args_str
    # TabPFN flags must NOT appear when tabular_model=xgb
    assert "--tabpfn_n_estimators" not in args_str


def test_build_trial_args_gkg_tabpfn_only_when_active():
    """gkg_tabpfn_n_estimators is only passed when use_gkg_change_detector=true AND gkg_change_model=tabpfn."""
    search_space = {
        "tabular_model": {"type": "categorical", "choices": ["xgb"]},
        "tabpfn_n_estimators": {"type": "int", "low": 4, "high": 16},
        "tabpfn_softmax_temperature": {"type": "float", "low": 0.5, "high": 1.5},
        "xgb_max_depth": {"type": "int", "low": 3, "high": 8},
        "xgb_learning_rate": {"type": "float", "low": 0.01, "high": 0.2},
        "use_blend": {"type": "categorical", "choices": ["false"]},
        "use_gkg_change_detector": {"type": "categorical", "choices": ["true"]},
        "gkg_change_model": {"type": "categorical", "choices": ["tabpfn"]},
        "gkg_tabpfn_n_estimators": {"type": "int", "low": 4, "high": 16},
        "chain_variants_config": {"type": "categorical", "choices": ["config/chain_variants.yaml"]},
    }
    params = {
        "tabular_model": "xgb",
        "use_gkg_change_detector": "true",
        "gkg_change_model": "tabpfn",
        "gkg_tabpfn_n_estimators": 12,
        "xgb_max_depth": 4,
        "xgb_learning_rate": 0.05,
        "use_blend": "false",
        "tabpfn_n_estimators": 8,
        "tabpfn_softmax_temperature": 0.9,
        "chain_variants_config": "config/chain_variants.yaml",
    }
    trial = _make_trial(params)
    args = sweep_runner.build_trial_args(trial, search_space, "^GSPC", [])
    args_str = " ".join(args)
    assert "--use_gkg_change_detector" in args_str
    assert "--gkg_change_model tabpfn" in args_str
    assert "--gkg_tabpfn_n_estimators 12" in args_str


# ---------------------------------------------------------------------------
# read_robust_score_from_mlflow
# ---------------------------------------------------------------------------

def test_read_robust_score_returns_score():
    mock_runs = pd.DataFrame([{"metrics.robust_score": 0.72}])
    with patch("sweep_runner.mlflow") as mock_mlflow:
        mock_mlflow.search_runs.return_value = mock_runs
        score = sweep_runner.read_robust_score_from_mlflow(
            "walk_fordward_sweep_adv", "^GSPC", 3, "sweep_GSPC_20260315"
        )
    assert score == pytest.approx(0.72)


def test_read_robust_score_returns_neg_inf_on_empty():
    with patch("sweep_runner.mlflow") as mock_mlflow:
        mock_mlflow.search_runs.return_value = pd.DataFrame()
        score = sweep_runner.read_robust_score_from_mlflow(
            "walk_fordward_sweep_adv", "^GSPC", 3, "sweep_GSPC_20260315"
        )
    assert score == float("-inf")


def test_read_robust_score_returns_neg_inf_on_nan():
    mock_runs = pd.DataFrame([{"metrics.robust_score": float("nan")}])
    with patch("sweep_runner.mlflow") as mock_mlflow:
        mock_mlflow.search_runs.return_value = mock_runs
        score = sweep_runner.read_robust_score_from_mlflow(
            "walk_fordward_sweep_adv", "^GSPC", 3, "sweep_GSPC_20260315"
        )
    assert score == float("-inf")
```

- [ ] **Step 2: Run to verify tests fail**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_sweep_runner.py -v
```

Expected: all FAILED — `sweep_runner` module does not exist.

- [ ] **Step 3: Install optuna**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry add optuna
```

- [ ] **Step 4: Create `scripts/sweep_runner.py`**

Create `scripts/sweep_runner.py` with this full implementation:

```python
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
import subprocess
import sys
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
    else:
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

    import tempfile, os
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
```

- [ ] **Step 5: Run unit tests to verify they pass**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_sweep_runner.py -v
```

Expected: all 8 tests PASSED.

- [ ] **Step 6: Run full test suite**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && git add scripts/sweep_runner.py tests/test_sweep_runner.py && git commit -m "feat(sweep): add sweep_runner.py with Optuna TPE + MLflow integration"
```
