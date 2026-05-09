# TabPFN for GKG Change Detector — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `NotImplementedError` stub in `gkg_change_detector.py` with a working TabPFN model path that mirrors the existing XGB branch and skips calibration automatically.

**Architecture:** Three targeted edits to one file: add 3 config fields, implement the `tabpfn` branch in `make_model()`, and replace the calibration block in `fit()` with a direct assignment when `model_type=tabpfn`. All other code paths are untouched.

**Tech Stack:** Python, TabPFN (`tabpfn.TabPFNClassifier`), pytest, unittest.mock

---

## Chunk 1: Config fields + make_model tabpfn branch

### Task 1: TabPFN config defaults

**Files:**
- Create: `tests/test_gkg_change_detector_tabpfn.py`
- Modify: `src/quant_risk/models/tabular/gkg_change_detector.py:40-54`

**Spec ref:** `docs/superpowers/specs/2026-03-15-tabpfn-gkg-detector-design.md` — "GkgChangeDetectorConfig — new fields"

- [ ] **Step 1: Write the failing test**

Create `tests/test_gkg_change_detector_tabpfn.py` with exactly this content:

```python
"""Tests for TabPFN support in the GKG change detector."""
import sys
import unittest.mock as mock

import numpy as np
import pytest

from quant_risk.models.tabular.gkg_change_detector import (
    GkgChangeDetectorConfig,
    fit,
    make_model,
    predict_proba,
)


def test_tabpfn_config_defaults():
    cfg = GkgChangeDetectorConfig(model_type="tabpfn")
    assert cfg.tabpfn_n_estimators == 8
    assert cfg.tabpfn_softmax_temperature == pytest.approx(0.9)
    assert cfg.tabpfn_balance_probabilities is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_gkg_change_detector_tabpfn.py::test_tabpfn_config_defaults -v
```

Expected: `FAILED` — `AttributeError: ... has no attribute 'tabpfn_n_estimators'`

- [ ] **Step 3: Add the three new fields to `GkgChangeDetectorConfig`**

In `src/quant_risk/models/tabular/gkg_change_detector.py`, find the `GkgChangeDetectorConfig` dataclass. Make these two changes:

**Change A** — update the `model_type` comment on line 42:
```python
    model_type: str = "logit"  # logit | xgb | tabpfn
```
(Remove `(placeholder)` — was `# logit | xgb | tabpfn(placeholder)`)

**Change B** — add after the `xgb_n_jobs` field (currently last field at line 54):
```python
    # TabPFN params (used only when model_type=tabpfn)
    tabpfn_n_estimators: int = 8
    tabpfn_softmax_temperature: float = 0.9
    tabpfn_balance_probabilities: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_gkg_change_detector_tabpfn.py::test_tabpfn_config_defaults -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && git add tests/test_gkg_change_detector_tabpfn.py src/quant_risk/models/tabular/gkg_change_detector.py && git commit -m "feat(gkg-detector): add TabPFN config fields"
```

---

### Task 2: make_model tabpfn branch

**Files:**
- Modify: `tests/test_gkg_change_detector_tabpfn.py` (add test)
- Modify: `src/quant_risk/models/tabular/gkg_change_detector.py:125-128`

**Spec ref:** "make_model() — tabpfn branch"

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gkg_change_detector_tabpfn.py`:

```python
def test_make_model_tabpfn_raises_import_error():
    with mock.patch.dict(sys.modules, {"tabpfn": None}):
        with pytest.raises(ImportError, match="tabpfn no está instalado"):
            make_model(GkgChangeDetectorConfig(model_type="tabpfn"))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_gkg_change_detector_tabpfn.py::test_make_model_tabpfn_raises_import_error -v
```

Expected: `FAILED` — the current code raises `NotImplementedError`, not `ImportError`

- [ ] **Step 3: Implement the tabpfn branch in make_model()**

In `src/quant_risk/models/tabular/gkg_change_detector.py`, find the block at lines 125–128:

```python
    if model_type == "tabpfn":
        raise NotImplementedError(
            "model_type='tabpfn' is reserved for future extension in the change detector."
        )
```

Replace it with:

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

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_gkg_change_detector_tabpfn.py::test_make_model_tabpfn_raises_import_error -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && git add tests/test_gkg_change_detector_tabpfn.py src/quant_risk/models/tabular/gkg_change_detector.py && git commit -m "feat(gkg-detector): implement tabpfn branch in make_model"
```

---

## Chunk 2: fit() calibration skip + integration test

### Task 3: Calibration skip in fit() + end-to-end test

**Files:**
- Modify: `tests/test_gkg_change_detector_tabpfn.py` (add 2 tests)
- Modify: `src/quant_risk/models/tabular/gkg_change_detector.py:207-214`

**Spec ref:** "fit() — skip calibration for tabpfn"

- [ ] **Step 1: Write the two failing tests**

Append to `tests/test_gkg_change_detector_tabpfn.py`:

```python
@pytest.mark.parametrize("cal_method", ["none", "platt", "isotonic"])
def test_fit_tabpfn_skips_calibration(cal_method):
    cfg = GkgChangeDetectorConfig(
        model_type="tabpfn",
        calibration_method=cal_method,
    )
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 5)).astype(np.float32)
    y = np.array([0, 1] * 15)
    feature_names = [f"f{i}" for i in range(5)]

    mock_clf = mock.MagicMock()
    mock_clf.predict_proba.return_value = np.column_stack(
        [np.full(30, 0.4), np.full(30, 0.6)]
    )

    with mock.patch("tabpfn.TabPFNClassifier", return_value=mock_clf):
        artifacts = fit(cfg, X, y, feature_names=feature_names)

    assert artifacts.calibrator["method"] == "none"
    assert artifacts.calibrator["model"] is None


def test_fit_tabpfn_produces_valid_probabilities():
    cfg = GkgChangeDetectorConfig(model_type="tabpfn", calibration_method="platt")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6)).astype(np.float32)
    y = np.array([0, 1] * 20)
    feature_names = [f"feat_{i}" for i in range(6)]

    n_rows = X.shape[0]
    raw_proba = np.column_stack(
        [rng.uniform(0.2, 0.5, n_rows), rng.uniform(0.5, 0.8, n_rows)]
    ).astype(np.float64)

    mock_clf = mock.MagicMock()
    mock_clf.predict_proba.return_value = raw_proba

    with mock.patch("tabpfn.TabPFNClassifier", return_value=mock_clf):
        artifacts = fit(cfg, X, y, feature_names=feature_names)
        proba = predict_proba(artifacts, X)

    # shape
    assert proba.shape == (n_rows, 2)
    # values in [0, 1]
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    # rows sum to ~1 — safe: predict_proba() reconstructs [1-p_change, p_change] from the
    # scalar in mock column 1; the mock's raw columns need not be normalised
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(n_rows), atol=1e-6)
    # calibrator unchanged (no calibration applied)
    assert artifacts.calibrator == {"method": "none", "model": None}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_gkg_change_detector_tabpfn.py::test_fit_tabpfn_skips_calibration tests/test_gkg_change_detector_tabpfn.py::test_fit_tabpfn_produces_valid_probabilities -v
```

Expected: `FAILED` — the current `fit()` calls `predict_proba_change` and `_fit_binary_calibrator` unconditionally, so with a mock the calibration path still runs and might return a non-"none" calibrator or raise.

- [ ] **Step 3: Implement the calibration skip in fit()**

In `src/quant_risk/models/tabular/gkg_change_detector.py`, find the `fit()` function. The section currently reads (lines 207–214):

```python
    model = make_model(cfg)
    model.fit(x, y)
    p_train = predict_proba_change(model, x)
    calibrator = _fit_binary_calibrator(
        p_change=p_train,
        y_true=y,
        method=str(cfg.calibration_method),
    )
```

Replace it with:

```python
    model = make_model(cfg)
    model.fit(x, y)
    if str(cfg.model_type).lower().strip() == "tabpfn":
        calibrator: dict = {"method": "none", "model": None}
    else:
        p_train = predict_proba_change(model, x)
        calibrator = _fit_binary_calibrator(
            p_change=p_train,
            y_true=y,
            method=str(cfg.calibration_method),
        )
```

- [ ] **Step 4: Run new tests to verify they pass**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest tests/test_gkg_change_detector_tabpfn.py -v
```

Expected: all 6 tests (1 + 1 + 3 parametrized + 1) `PASSED`

- [ ] **Step 5: Run the full test suite to verify nothing is broken**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && poetry run pytest --tb=short -q
```

Expected: all existing tests pass; total count increases by 6.

- [ ] **Step 6: Commit**

```bash
cd /home/manidmt/TFM/quant-risk-tfm && git add tests/test_gkg_change_detector_tabpfn.py src/quant_risk/models/tabular/gkg_change_detector.py && git commit -m "feat(gkg-detector): skip calibration for tabpfn in fit()"
```
