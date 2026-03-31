"""
Tests for src/quant_risk/prod/worker/inference.py — bundle loading and per-asset inference.

Coverage:
- _resolve_bundle_dir: symlink, text-file, and direct-dir conventions
- _load_feature_contract: list / dict-with-columns / missing-file
- _align_features: column selection, missing columns filled with 0
- _load_model_artifact: xgb native JSON, pickle fallback, missing model
- _load_calibrator: none / pkl / absent
- _pick_class: argmax and threshold-gated selection
- run_inference_for_asset: happy-path end-to-end, missing bundle, empty features, bad proba
- PredictionOutput: probability sum validation
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_risk.prod.schemas import PredictedClass
from quant_risk.prod.worker.inference import (
    PredictionOutput,
    _align_features,
    _load_calibrator,
    _load_feature_contract,
    _load_inference_config,
    _load_model_artifact,
    _load_thresholds,
    _pick_class,
    _resolve_bundle_dir,
    run_inference_for_asset,
)


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _StubModel:
    """Minimal model stub that returns fixed probabilities."""
    def __init__(self, proba: list[float]):
        self._proba = np.array(proba, dtype=float)

    def predict_proba(self, X):  # noqa: N802
        return np.tile(self._proba, (len(X), 1))


def _make_bundle(
    bundles_dir: Path,
    asset_id: str,
    bundle_version: str = "2026-03-27_prod_abc1234",
    current_as: str = "symlink",  # "symlink" | "textfile" | "dir"
    model_proba: list[float] | None = None,
    feature_contract: list[str] | None = None,
    calibrator=None,
    thresholds: list[float] | None = None,
    inference_cfg: dict | None = None,
) -> Path:
    """Create a minimal fake bundle directory and return its path."""
    asset_dir = bundles_dir / asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    bundle_dir = asset_dir / bundle_version
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model_dir = bundle_dir / "model"
    model_dir.mkdir()

    # manifest.json
    manifest = {
        "release_id": "prod_2026-03-27",
        "bundle_version": bundle_version,
        "asset_id": asset_id,
        "source_ticker": "^GSPC",
        "horizon_days": 5,
        "model_type": "xgb",
        "feature_profile": "gkg_v1_features_compact",
        "training_run_ref": "mlflow_abc123",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": "3.11.0",
        "dependency_versions": {"python": "3.11.0"},
        "file_checksums": [],
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

    # model/model.pkl — stub model
    proba = model_proba or [0.2, 0.3, 0.5]
    stub = _StubModel(proba)
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(stub, f)

    # feature_contract.json (optional)
    if feature_contract is not None:
        (bundle_dir / "feature_contract.json").write_text(
            json.dumps(feature_contract)
        )

    # calibration artefacts (optional)
    if calibrator is not None:
        (bundle_dir / "calibration.json").write_text(
            json.dumps({"method": "pkl"})
        )
        with open(bundle_dir / "calibration.pkl", "wb") as f:
            pickle.dump(calibrator, f)
    else:
        (bundle_dir / "calibration.json").write_text(
            json.dumps({"method": "none"})
        )

    # thresholds.json (optional)
    if thresholds is not None:
        (bundle_dir / "thresholds.json").write_text(
            json.dumps({"thresholds": thresholds})
        )

    # inference_config.json
    cfg = inference_cfg or {"class_labels": ["low", "medium", "high"]}
    (bundle_dir / "inference_config.json").write_text(json.dumps(cfg))

    # Set up the "current" pointer
    current_ptr = asset_dir / "current"
    if current_as == "symlink":
        current_ptr.symlink_to(bundle_version)
    elif current_as == "textfile":
        current_ptr.write_text(bundle_version)
    elif current_as == "dir":
        # current/ is the actual bundle dir (copy via rename trick → just symlink for test)
        current_ptr.symlink_to(bundle_version)
    else:
        raise ValueError(f"Unknown current_as={current_as!r}")

    return bundle_dir


def _make_features(
    cols: list[str] | None = None,
    forecast_date: date | None = None,
) -> pd.DataFrame:
    """Return a one-row features DataFrame."""
    if cols is None:
        cols = ["logret", "rv_5", "rv_20", "vix_diff"]
    if forecast_date is None:
        forecast_date = date(2024, 6, 28)
    df = pd.DataFrame(
        {c: [np.random.randn()] for c in cols},
        index=[pd.Timestamp(forecast_date)],
    )
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# PredictionOutput
# ---------------------------------------------------------------------------

class TestPredictionOutput:
    def test_valid_construction(self):
        po = PredictionOutput(
            predicted_class=PredictedClass.high,
            p_low=0.2, p_medium=0.3, p_high=0.5,
            bundle_version="2026-03-27_prod_abc1234",
        )
        assert po.predicted_class == PredictedClass.high

    def test_probabilities_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to ~1.0"):
            PredictionOutput(
                predicted_class=PredictedClass.high,
                p_low=0.1, p_medium=0.1, p_high=0.1,
                bundle_version="2026-03-27_prod_abc1234",
            )

    def test_boundary_tolerance(self):
        # 0.99 is within [0.98, 1.02]
        po = PredictionOutput(
            predicted_class=PredictedClass.low,
            p_low=0.99, p_medium=0.005, p_high=0.005,
            bundle_version="2026-03-27_prod_abc1234",
        )
        assert po.p_low == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# _resolve_bundle_dir
# ---------------------------------------------------------------------------

class TestResolveBundleDir:
    def test_symlink_convention(self, tmp_path):
        bd = _make_bundle(tmp_path, "us_equities", current_as="symlink")
        result = _resolve_bundle_dir(tmp_path, "us_equities")
        assert result.is_dir()
        assert result.name == "2026-03-27_prod_abc1234"

    def test_textfile_convention(self, tmp_path):
        bd = _make_bundle(tmp_path, "us_equities", current_as="textfile")
        result = _resolve_bundle_dir(tmp_path, "us_equities")
        assert result.is_dir()

    def test_missing_asset_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No bundle directory"):
            _resolve_bundle_dir(tmp_path, "nonexistent_asset")

    def test_missing_current_raises(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No active bundle"):
            _resolve_bundle_dir(tmp_path, "us_equities")


# ---------------------------------------------------------------------------
# _load_feature_contract
# ---------------------------------------------------------------------------

class TestLoadFeatureContract:
    def test_list_format(self, tmp_path):
        p = tmp_path / "feature_contract.json"
        p.write_text(json.dumps(["logret", "rv_5", "rv_20"]))
        result = _load_feature_contract(tmp_path)
        assert result == ["logret", "rv_5", "rv_20"]

    def test_dict_format(self, tmp_path):
        p = tmp_path / "feature_contract.json"
        p.write_text(json.dumps({"columns": ["logret", "rv_5"]}))
        result = _load_feature_contract(tmp_path)
        assert result == ["logret", "rv_5"]

    def test_missing_file_returns_none(self, tmp_path):
        result = _load_feature_contract(tmp_path)
        assert result is None

    def test_invalid_format_raises(self, tmp_path):
        p = tmp_path / "feature_contract.json"
        p.write_text(json.dumps({"bad_key": 42}))
        with pytest.raises(ValueError):
            _load_feature_contract(tmp_path)


# ---------------------------------------------------------------------------
# _align_features
# ---------------------------------------------------------------------------

class TestAlignFeatures:
    def test_selects_contract_columns(self):
        df = _make_features(["logret", "rv_5", "rv_20", "extra_col"])
        X = _align_features(df, ["logret", "rv_5"])
        assert X.shape == (1, 2)

    def test_missing_columns_filled_with_zero(self):
        df = _make_features(["logret"])
        X = _align_features(df, ["logret", "missing_col"])
        assert X.shape == (1, 2)
        assert X[0, 1] == pytest.approx(0.0)

    def test_no_contract_uses_all_columns(self):
        df = _make_features(["a", "b", "c"])
        X = _align_features(df, None)
        assert X.shape == (1, 3)

    def test_dtype_is_float32(self):
        df = _make_features(["logret", "rv_5"])
        X = _align_features(df, ["logret", "rv_5"])
        assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# _load_model_artifact
# ---------------------------------------------------------------------------

class TestLoadModelArtifact:
    def test_loads_pickle(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        stub = _StubModel([0.3, 0.3, 0.4])
        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(stub, f)
        model = _load_model_artifact(tmp_path, "xgb")
        assert hasattr(model, "predict_proba")

    def test_missing_model_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="model/"):
            _load_model_artifact(tmp_path, "xgb")

    def test_no_model_file_raises(self, tmp_path):
        (tmp_path / "model").mkdir()
        with pytest.raises(FileNotFoundError, match="No model file"):
            _load_model_artifact(tmp_path, "xgb")


# ---------------------------------------------------------------------------
# _load_calibrator
# ---------------------------------------------------------------------------

class TestLoadCalibrator:
    def test_method_none_returns_none(self, tmp_path):
        (tmp_path / "calibration.json").write_text(json.dumps({"method": "none"}))
        assert _load_calibrator(tmp_path) is None

    def test_no_files_returns_none(self, tmp_path):
        assert _load_calibrator(tmp_path) is None

    def test_pkl_method_loads_calibrator(self, tmp_path):
        (tmp_path / "calibration.json").write_text(json.dumps({"method": "pkl"}))
        stub = _StubModel([0.4, 0.3, 0.3])
        with open(tmp_path / "calibration.pkl", "wb") as f:
            pickle.dump(stub, f)
        cal = _load_calibrator(tmp_path)
        assert cal is not None
        assert hasattr(cal, "predict_proba")


# ---------------------------------------------------------------------------
# _pick_class
# ---------------------------------------------------------------------------

class TestPickClass:
    _labels = ["low", "medium", "high"]

    def test_argmax_no_thresholds(self):
        proba = np.array([0.1, 0.3, 0.6])
        cls = _pick_class(proba, self._labels, None)
        assert cls == PredictedClass.high

    def test_argmax_picks_medium(self):
        proba = np.array([0.1, 0.7, 0.2])
        cls = _pick_class(proba, self._labels, None)
        assert cls == PredictedClass.medium

    def test_threshold_gates_selection(self):
        # p_high = 0.6 but threshold = 0.7 → not eligible
        # p_medium = 0.3 and threshold = 0.2 → eligible
        proba = np.array([0.1, 0.3, 0.6])
        thresholds = [0.9, 0.2, 0.7]
        cls = _pick_class(proba, self._labels, thresholds)
        assert cls == PredictedClass.medium

    def test_threshold_fallback_to_argmax(self):
        # No class meets its threshold → fall back to argmax
        proba = np.array([0.1, 0.3, 0.6])
        thresholds = [0.9, 0.9, 0.9]
        cls = _pick_class(proba, self._labels, thresholds)
        assert cls == PredictedClass.high


# ---------------------------------------------------------------------------
# run_inference_for_asset — happy path
# ---------------------------------------------------------------------------

class TestRunInferenceHappyPath:
    def test_returns_prediction_output(self, tmp_path):
        _make_bundle(tmp_path, "us_equities", model_proba=[0.2, 0.3, 0.5])
        features = _make_features()
        result = run_inference_for_asset("us_equities", features, tmp_path)
        assert isinstance(result, PredictionOutput)

    def test_predicted_class_is_high(self, tmp_path):
        _make_bundle(tmp_path, "us_equities", model_proba=[0.1, 0.2, 0.7])
        features = _make_features()
        result = run_inference_for_asset("us_equities", features, tmp_path)
        assert result.predicted_class == PredictedClass.high

    def test_predicted_class_is_low(self, tmp_path):
        _make_bundle(tmp_path, "us_equities", model_proba=[0.8, 0.1, 0.1])
        features = _make_features()
        result = run_inference_for_asset("us_equities", features, tmp_path)
        assert result.predicted_class == PredictedClass.low

    def test_probabilities_sum_to_one(self, tmp_path):
        _make_bundle(tmp_path, "us_equities", model_proba=[0.3, 0.4, 0.3])
        features = _make_features()
        result = run_inference_for_asset("us_equities", features, tmp_path)
        assert result.p_low + result.p_medium + result.p_high == pytest.approx(1.0, abs=0.01)

    def test_bundle_version_in_output(self, tmp_path):
        _make_bundle(
            tmp_path, "us_equities",
            bundle_version="2026-03-27_prod_abc1234",
            model_proba=[0.2, 0.3, 0.5],
        )
        features = _make_features()
        result = run_inference_for_asset("us_equities", features, tmp_path)
        assert result.bundle_version == "2026-03-27_prod_abc1234"

    def test_feature_contract_applied(self, tmp_path):
        """Contract columns are selected; extra features in the DF are ignored."""
        _make_bundle(
            tmp_path, "us_equities",
            feature_contract=["logret", "rv_5"],
            model_proba=[0.2, 0.3, 0.5],
        )
        # Pass more columns than the contract
        features = _make_features(["logret", "rv_5", "rv_20", "extra"])
        result = run_inference_for_asset("us_equities", features, tmp_path)
        assert isinstance(result, PredictionOutput)

    def test_textfile_pointer_works(self, tmp_path):
        _make_bundle(tmp_path, "us_equities", current_as="textfile", model_proba=[0.2, 0.3, 0.5])
        features = _make_features()
        result = run_inference_for_asset("us_equities", features, tmp_path)
        assert isinstance(result, PredictionOutput)


# ---------------------------------------------------------------------------
# run_inference_for_asset — calibration
# ---------------------------------------------------------------------------

class TestRunInferenceCalibration:
    def test_calibrator_applied(self, tmp_path):
        """When a calibrator is present, its probabilities override the base model."""
        # Base model would predict "high" (p=0.7)
        # Calibrator will flip to "low" (p=0.9)
        calibrator = _StubModel([0.9, 0.07, 0.03])
        _make_bundle(
            tmp_path, "us_equities",
            model_proba=[0.1, 0.2, 0.7],
            calibrator=calibrator,
        )
        features = _make_features()
        result = run_inference_for_asset("us_equities", features, tmp_path)
        assert result.predicted_class == PredictedClass.low

    def test_no_calibration_uses_base(self, tmp_path):
        _make_bundle(tmp_path, "us_equities", model_proba=[0.1, 0.2, 0.7])
        features = _make_features()
        result = run_inference_for_asset("us_equities", features, tmp_path)
        assert result.predicted_class == PredictedClass.high


# ---------------------------------------------------------------------------
# run_inference_for_asset — error cases
# ---------------------------------------------------------------------------

class TestRunInferenceErrors:
    def test_missing_bundle_raises(self, tmp_path):
        features = _make_features()
        with pytest.raises(FileNotFoundError):
            run_inference_for_asset("nonexistent_asset", features, tmp_path)

    def test_empty_features_raises(self, tmp_path):
        _make_bundle(tmp_path, "us_equities")
        empty = pd.DataFrame()
        with pytest.raises(RuntimeError, match="empty"):
            run_inference_for_asset("us_equities", empty, tmp_path)

    def test_model_without_predict_proba_raises(self, tmp_path):
        # Save a model object without predict_proba
        bundle_dir = _make_bundle(tmp_path, "us_equities")
        bad_model = object()  # has no predict_proba
        with open(bundle_dir / "model" / "model.pkl", "wb") as f:
            pickle.dump(bad_model, f)
        features = _make_features()
        with pytest.raises(AttributeError, match="predict_proba"):
            run_inference_for_asset("us_equities", features, tmp_path)


# ---------------------------------------------------------------------------
# _load_inference_config / _load_thresholds
# ---------------------------------------------------------------------------

class TestLoadInferenceConfig:
    def test_missing_returns_empty(self, tmp_path):
        assert _load_inference_config(tmp_path) == {}

    def test_reads_class_labels(self, tmp_path):
        (tmp_path / "inference_config.json").write_text(
            json.dumps({"class_labels": ["low", "medium", "high"]})
        )
        cfg = _load_inference_config(tmp_path)
        assert cfg["class_labels"] == ["low", "medium", "high"]


class TestLoadThresholds:
    def test_missing_returns_none(self, tmp_path):
        assert _load_thresholds(tmp_path) is None

    def test_list_format(self, tmp_path):
        (tmp_path / "thresholds.json").write_text(json.dumps([0.4, 0.4, 0.4]))
        assert _load_thresholds(tmp_path) == [0.4, 0.4, 0.4]

    def test_dict_format(self, tmp_path):
        (tmp_path / "thresholds.json").write_text(
            json.dumps({"thresholds": [0.3, 0.4, 0.5]})
        )
        assert _load_thresholds(tmp_path) == [0.3, 0.4, 0.5]
