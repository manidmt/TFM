# TabPFN for GKG Change Detector — Design Spec

**Date**: 2026-03-15
**Branch**: feature/mlflow
**Scope**: Phase 1 only — TabPFN support in `gkg_change_detector.py`

---

## Goal

Implement the currently stubbed `model_type='tabpfn'` path in the GKG change detector so it can be used as a drop-in alternative to logit/XGB for regime-change classification.

---

## Context

The GKG change detector (`src/quant_risk/models/tabular/gkg_change_detector.py`) classifies each trading day as a regime change (1) or no-change (0) using news-derived features. It supports `logit` and `xgb` model types; the `tabpfn` branch currently raises `NotImplementedError`.

Training dataset characteristics per fold:
- ~750 rows (one per business day, GKG start 2020-01-01)
- ~111 features (GKG topic signals across 5 topics + optional context columns)
- ~46.5% positive rate — approximately balanced

TabPFN's practical row limit is ~10k. The dataset is well below this threshold; no subsampling is required.

---

## Scope

**In scope**:
- Add 3 new fields to `GkgChangeDetectorConfig`
- Implement `tabpfn` branch in `make_model()`
- Skip calibration in `fit()` when `model_type=tabpfn`
- Unit tests in `tests/test_gkg_change_detector_tabpfn.py`

**Out of scope**:
- Config-file-driven TabPFN defaults
- Shared TabPFN builder across main tabular model and change detector
- Any changes to `walk_forward_chain_tab.py` or other scripts
- Hyperparameter search for TabPFN in the change detector

---

## Changes to `src/quant_risk/models/tabular/gkg_change_detector.py`

### 1. `GkgChangeDetectorConfig` — new fields

```python
# TabPFN params (used only when model_type=tabpfn)
tabpfn_n_estimators: int = 8
tabpfn_softmax_temperature: float = 0.9
tabpfn_balance_probabilities: bool = False
```

Also update the `model_type` comment from `# logit | xgb | tabpfn(placeholder)` to `# logit | xgb | tabpfn`.

Rationale for defaults:
- `n_estimators=8`: fast enough for ~750 rows without sacrificing quality
- `softmax_temperature=0.9`: slight sharpening, consistent with main tabular TabPFN usage
- `balance_probabilities=False`: the dataset is ~46% positive, so forced balancing is not needed

`random_state` is already a field on `GkgChangeDetectorConfig` (used by all model types) and is passed to the TabPFN constructor (see section 2).

### 2. `make_model()` — tabpfn branch

Replace the `NotImplementedError` with:

```python
if model_type == "tabpfn":
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as e:
        raise ImportError(
            "tabpfn no está instalado. Instala con: poetry add tabpfn"
        ) from e
    set_global_seed(cfg.seed, use_torch=True)
    return TabPFNClassifier(
        n_estimators=cfg.tabpfn_n_estimators,
        softmax_temperature=cfg.tabpfn_softmax_temperature,
        balance_probabilities=cfg.tabpfn_balance_probabilities,
        random_state=int(cfg.random_state),
    )
```

The lazy import pattern mirrors the existing `xgb` branch.
`set_global_seed(..., use_torch=True)` is used because TabPFN relies on PyTorch.
`random_state=int(cfg.random_state)` uses an explicit cast for consistency with the `xgb` branch in the same file (line 123), even though `GkgChangeDetectorConfig.random_state` is already typed `int`.

### 3. `fit()` — skip calibration for tabpfn

TabPFN outputs well-calibrated probabilities by design. When `model_type == "tabpfn"`, the entire block that computes `p_train` and fits the calibrator (currently lines 209–213) is replaced by a single assignment:

```python
calibrator: dict = {"method": "none", "model": None}
```

That is: `predict_proba_change(model, x)` is **not** called and `_fit_binary_calibrator` is **not** called. The calibrator is set directly. `cfg.calibration_method` is silently ignored for `tabpfn`.

The non-tabpfn path (logit, xgb) is unchanged. All of `predict_proba()` / `predict_proba_change()` are unchanged.

**NaN sanitization**: No additional sanitization step is added for the tabpfn branch. The GKG design matrix is built by `build_design_matrix()` upstream, which produces a clean `float32` array. The existing logit and xgb paths also rely on this upstream guarantee and do not sanitize inside `fit()`. This is intentional.

**Calibration override warning**: When `cfg.calibration_method != "none"` and `model_type == "tabpfn"`, no warning is emitted. The override is silent, by design — consistent with how other ignored-but-valid config combinations behave in this codebase (e.g., passing `balance_probabilities=False` is not warned about for logit). Users setting `calibration_method` in a shared config and switching `model_type` to `tabpfn` should not see noise in their logs.

---

## Tests: `tests/test_gkg_change_detector_tabpfn.py`

| Test | What it verifies |
|---|---|
| `test_tabpfn_config_defaults` | New config fields have correct default values |
| `test_make_model_tabpfn_raises_import_error` | Missing `tabpfn` package gives clear ImportError message |
| `test_fit_tabpfn_skips_calibration` | Parameterized over `calibration_method` in `{"none", "platt", "isotonic"}`: `calibrator["method"] == "none"` in all three cases |
| `test_fit_tabpfn_produces_valid_probabilities` | fit + predict on small synthetic dataset; output shape correct, values in [0, 1], and `artifacts.calibrator == {"method": "none", "model": None}` |

**Mocking strategy**: `TabPFNClassifier` is introduced by a local `from tabpfn import TabPFNClassifier` inside `make_model()`. Because it is never bound into the `gkg_change_detector` module's namespace, patching `"quant_risk.models.tabular.gkg_change_detector.TabPFNClassifier"` would have no effect. The correct patch target is the source module:

```python
unittest.mock.patch("tabpfn.TabPFNClassifier")
```

Use this target in `test_fit_tabpfn_skips_calibration` and `test_fit_tabpfn_produces_valid_probabilities`. `test_tabpfn_config_defaults` only checks dataclass fields and does not call `make_model()`, so no mock is needed there. This intercepts the class on the `tabpfn` module before it is imported locally, preventing any real PyTorch/GPU initialisation.

For `test_make_model_tabpfn_raises_import_error`, since `tabpfn` IS installed in the project, the `ImportError` branch is reached by patching the module out of `sys.modules`:

```python
import sys
import unittest.mock as mock

def test_make_model_tabpfn_raises_import_error():
    with mock.patch.dict(sys.modules, {"tabpfn": None}):
        with pytest.raises(ImportError, match="tabpfn no está instalado"):
            make_model(GkgChangeDetectorConfig(model_type="tabpfn"))
```

Setting `sys.modules["tabpfn"] = None` causes `from tabpfn import TabPFNClassifier` to raise `ImportError` inside `make_model()`, covering the error branch without uninstalling the package.

---

## Failure Policy

| Condition | Behaviour |
|---|---|
| `tabpfn` not installed | `ImportError` with install instruction (same as xgb branch) |
| `calibration_method` set with `model_type=tabpfn` | Silently ignored; "none" is always used. No warning is emitted (by design). |
| `model_type` not in {logit, xgb, tabpfn} | `ValueError` (existing behaviour, unchanged) |

---

## Acceptance Criteria

- `make_model(GkgChangeDetectorConfig(model_type="tabpfn"))` returns a `TabPFNClassifier`
- `fit()` with `model_type=tabpfn` stores `calibrator={"method": "none", "model": None}` regardless of `cfg.calibration_method`
- All existing tests continue to pass
- New `tests/test_gkg_change_detector_tabpfn.py` passes

---

## Non-Goals

- Do not add a config YAML for TabPFN GKG defaults
- Do not create a shared TabPFN factory with the main tabular model
- Do not modify any script outside `gkg_change_detector.py`
