'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Production bundle loading and per-asset inference.

Loads the active bundle for an asset from the bundles/ directory tree and runs
5-day volatility regime inference.

Bundle layout expected (rpi5.md §8.1):

    bundles/
      <asset_id>/
        current -> <bundle_version>/   (symlink or plain text file with version name)
        <bundle_version>/
          manifest.json
          model/
            model.json       ← XGBoost native format (preferred)
            model.pkl        ← generic pickle fallback
          feature_contract.json   ← list[str] of expected feature columns
          inference_config.json   ← {"class_labels": ["low","medium","high"], ...}
          calibration.json        ← {"method": "none"} or {"method": "pkl"}
          calibration.pkl         ← sklearn calibration object (if method="pkl")
          thresholds.json         ← {"thresholds": [t_low, t_medium, t_high]} (optional)

Model type dispatch (manifest.model_type):
  - "xgb"    → XGBClassifier loaded from model/model.json or model/model.pkl
  - "tabpfn" → object loaded from model/model.pkl
  - others   → generic pickle/joblib fallback

Calibration:
  If calibration.pkl exists and calibration.json specifies method="pkl", the
  pickle is loaded and called as calibrator.predict_proba(X).  Otherwise raw
  probabilities from the base model are used directly.

Reference: rpi5.md §8, §9, §10.3 (steps 5-6)
'''

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_risk.prod.bundle_manifest import BundleManifest
from quant_risk.prod.schemas import PredictedClass

logger = logging.getLogger(__name__)

# Default class label ordering: index 0 = low, 1 = medium, 2 = high
_DEFAULT_CLASS_LABELS = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PredictionOutput:
    """Inference result for a single asset on a single forecast date.

    Attributes
    ----------
    predicted_class:
        The winning volatility regime class.
    p_low / p_medium / p_high:
        Calibrated class probabilities.  Must sum to ~1.0.
    bundle_version:
        The bundle version that produced this prediction.
    """

    predicted_class: PredictedClass
    p_low: float
    p_medium: float
    p_high: float
    bundle_version: str

    def __post_init__(self) -> None:
        total = self.p_low + self.p_medium + self.p_high
        if not (0.98 <= total <= 1.02):
            raise ValueError(
                f"Class probabilities must sum to ~1.0, got {total:.4f} "
                f"(p_low={self.p_low}, p_medium={self.p_medium}, p_high={self.p_high})."
            )


# ---------------------------------------------------------------------------
# Bundle path resolution
# ---------------------------------------------------------------------------

def _resolve_bundle_dir(bundles_dir: str | Path, asset_id: str) -> Path:
    """Return the active bundle directory for *asset_id*.

    Supports two ``current`` pointer conventions:
    - symbolic link pointing to the bundle version directory
    - plain text file containing the bundle version name

    Raises
    ------
    FileNotFoundError
        If the asset directory or ``current`` pointer is missing.
    """
    asset_dir = Path(bundles_dir) / asset_id
    if not asset_dir.exists():
        raise FileNotFoundError(
            f"No bundle directory for asset '{asset_id}': '{asset_dir}' not found."
        )

    current_ptr = asset_dir / "current"
    if not current_ptr.exists():
        raise FileNotFoundError(
            f"No active bundle for asset '{asset_id}': "
            f"'{current_ptr}' (symlink or version pointer) not found."
        )

    if current_ptr.is_symlink():
        bundle_dir = current_ptr.resolve()
    elif current_ptr.is_dir():
        # current/ is itself the bundle directory (development shortcut)
        bundle_dir = current_ptr.resolve()
    else:
        # Plain text file containing the bundle version name
        bundle_version = current_ptr.read_text(encoding="utf-8").strip()
        if not bundle_version:
            raise ValueError(
                f"'current' pointer file for asset '{asset_id}' is empty."
            )
        bundle_dir = asset_dir / bundle_version

    if not bundle_dir.is_dir():
        raise FileNotFoundError(
            f"Active bundle directory '{bundle_dir}' for asset '{asset_id}' not found."
        )
    return bundle_dir


# ---------------------------------------------------------------------------
# Bundle artefact loaders
# ---------------------------------------------------------------------------

def _load_feature_contract(bundle_dir: Path) -> list[str] | None:
    """Return the feature column list from feature_contract.json, or None if absent."""
    path = bundle_dir / "feature_contract.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(c) for c in data]
    if isinstance(data, dict) and "columns" in data:
        return [str(c) for c in data["columns"]]
    raise ValueError(
        f"feature_contract.json must be a list of column names or "
        f"a dict with a 'columns' key. Got type {type(data).__name__}."
    )


def _load_inference_config(bundle_dir: Path) -> dict[str, Any]:
    """Return parsed inference_config.json, or an empty dict if the file is absent."""
    path = bundle_dir / "inference_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_thresholds(bundle_dir: Path) -> list[float] | None:
    """Return per-class thresholds from thresholds.json, or None."""
    path = bundle_dir / "thresholds.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [float(t) for t in data]
    if isinstance(data, dict) and "thresholds" in data:
        return [float(t) for t in data["thresholds"]]
    return None


def _load_model_artifact(bundle_dir: Path, model_type: str) -> Any:
    """Load the trained model from bundle_dir/model/.

    Dispatch order:
    1. XGBoost native JSON (model.json) for model_type="xgb"
    2. Pickle file (model.pkl) for all types
    3. Joblib file (model.joblib) as generic fallback

    Raises
    ------
    FileNotFoundError
        If no recognisable model file is found.
    """
    model_dir = bundle_dir / "model"
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"model/ sub-directory not found inside bundle '{bundle_dir}'."
        )

    mt = str(model_type).lower()

    if mt == "xgb":
        native = model_dir / "model.json"
        if native.exists():
            try:
                from xgboost import XGBClassifier  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "xgboost is required for model_type='xgb'. "
                    "Install with: poetry add xgboost"
                ) from exc
            model = XGBClassifier()
            model.load_model(str(native))
            logger.debug("Loaded XGBoost model from %s", native)
            return model

    # Generic pickle fallback (works for xgb, tabpfn, sklearn, etc.)
    pkl = model_dir / "model.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            model = pickle.load(f)
        logger.debug("Loaded model from pickle %s", pkl)
        return model

    # Joblib fallback
    joblib_path = model_dir / "model.joblib"
    if joblib_path.exists():
        try:
            import joblib  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("joblib is required to load model.joblib.") from exc
        model = joblib.load(str(joblib_path))
        logger.debug("Loaded model from joblib %s", joblib_path)
        return model

    raise FileNotFoundError(
        f"No model file found in '{model_dir}' for model_type='{model_type}'. "
        "Expected one of: model.json, model.pkl, model.joblib."
    )


def _load_calibrator(bundle_dir: Path) -> Any | None:
    """Load the probability calibrator from calibration.pkl if indicated.

    Returns None if no calibration is configured or the file is absent.
    """
    cal_json = bundle_dir / "calibration.json"
    cal_pkl = bundle_dir / "calibration.pkl"

    if not cal_json.exists() and not cal_pkl.exists():
        return None

    if cal_json.exists():
        meta = json.loads(cal_json.read_text(encoding="utf-8"))
        method = str(meta.get("method", "none")).lower()
        if method == "none":
            return None
        if method != "pkl":
            logger.warning(
                "Unsupported calibration method '%s' in bundle '%s'; skipping calibration.",
                method, bundle_dir,
            )
            return None

    if cal_pkl.exists():
        with open(cal_pkl, "rb") as f:
            calibrator = pickle.load(f)
        logger.debug("Loaded calibrator from %s", cal_pkl)
        return calibrator

    return None


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------

def _align_features(
    features: pd.DataFrame,
    feature_contract: list[str] | None,
) -> np.ndarray:
    """Return a (1, n_features) float32 array aligned to *feature_contract*.

    Missing contract columns are filled with 0.0.  If *feature_contract* is
    None, all columns in *features* are used in their current order.
    """
    if feature_contract is None:
        return features.to_numpy(dtype=np.float32)

    aligned = pd.DataFrame(index=features.index)
    for col in feature_contract:
        if col in features.columns:
            aligned[col] = features[col]
        else:
            aligned[col] = 0.0

    return aligned.to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def _raw_predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Return (1, n_classes) probability array from any model supporting predict_proba."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"Model of type '{type(model).__name__}' does not implement predict_proba. "
            "Production inference requires calibrated class probabilities."
        )
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)
    return proba


def _pick_class(
    proba_row: np.ndarray,
    class_labels: list[str],
    thresholds: list[float] | None,
) -> PredictedClass:
    """Select the predicted class from a probability vector.

    If *thresholds* is provided, uses the highest-probability class that
    exceeds its threshold; falls back to argmax if none exceed their threshold.
    """
    if thresholds and len(thresholds) == len(class_labels):
        # Threshold-gated selection: pick class with highest p that beats its threshold
        eligible = [
            (i, p) for i, (p, t) in enumerate(zip(proba_row, thresholds)) if p >= t
        ]
        if eligible:
            best_idx = max(eligible, key=lambda x: x[1])[0]
        else:
            best_idx = int(np.argmax(proba_row))
    else:
        best_idx = int(np.argmax(proba_row))

    label = class_labels[best_idx] if best_idx < len(class_labels) else "medium"
    return PredictedClass(label)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_inference_for_asset(
    asset_id: str,
    features: pd.DataFrame,
    bundles_dir: str | Path,
) -> PredictionOutput:
    """Load the active bundle and run 5-day inference for one asset.

    Steps:
    1. Resolve ``bundles_dir/<asset_id>/current`` → bundle directory
    2. Load and log manifest (validates bundle integrity metadata)
    3. Load feature_contract.json → align feature columns
    4. Load model artifact (XGB native JSON → pickle fallback)
    5. Load calibrator (calibration.pkl if method="pkl")
    6. Run predict_proba; apply calibrator if present
    7. Select predicted class (argmax or threshold-gated)
    8. Return PredictionOutput

    Parameters
    ----------
    asset_id:
        Semantic asset identifier (used to locate the bundle directory).
    features:
        Single-row DataFrame produced by build_features_for_asset.
    bundles_dir:
        Root directory of the bundle tree (``/srv/quant-risk/bundles`` on RPi5).

    Returns
    -------
    PredictionOutput
        Calibrated class probabilities and the winning class.

    Raises
    ------
    FileNotFoundError
        If the active bundle cannot be located.
    RuntimeError
        If features are empty or probability extraction fails.
    """
    if features.empty:
        raise RuntimeError(
            f"[{asset_id}] Cannot run inference: features DataFrame is empty."
        )

    # 1. Resolve active bundle directory
    bundle_dir = _resolve_bundle_dir(bundles_dir, asset_id)
    logger.debug("[%s] Active bundle: %s", asset_id, bundle_dir)

    # 2. Load manifest
    manifest = BundleManifest.load(bundle_dir)
    logger.debug(
        "[%s] Loaded manifest — version=%s, model_type=%s, horizon=%d",
        asset_id, manifest.bundle_version, manifest.model_type, manifest.horizon_days,
    )

    # 3. Load supporting artefacts
    feature_contract = _load_feature_contract(bundle_dir)
    inference_cfg = _load_inference_config(bundle_dir)
    thresholds = _load_thresholds(bundle_dir)
    class_labels: list[str] = inference_cfg.get("class_labels", _DEFAULT_CLASS_LABELS)

    # 4. Align features to contract
    X = _align_features(features, feature_contract)

    # 5. Load model
    model = _load_model_artifact(bundle_dir, manifest.model_type)

    # 6. Get raw probabilities
    proba = _raw_predict_proba(model, X)   # shape (1, n_classes)
    proba_row = proba[0]

    # 7. Apply calibrator (if any)
    calibrator = _load_calibrator(bundle_dir)
    if calibrator is not None:
        cal_proba = np.asarray(calibrator.predict_proba(X), dtype=float)
        if cal_proba.ndim == 1:
            cal_proba = cal_proba.reshape(1, -1)
        proba_row = cal_proba[0]
        logger.debug("[%s] Applied calibration — proba=%s", asset_id, proba_row)

    # Normalise to sum=1 (guard against floating-point drift)
    total = float(proba_row.sum())
    if total > 0:
        proba_row = proba_row / total
    else:
        proba_row = np.array([1 / len(class_labels)] * len(class_labels))

    # Ensure we have exactly 3 class probabilities
    if len(proba_row) != 3:
        raise RuntimeError(
            f"[{asset_id}] Expected 3-class probability vector, got length {len(proba_row)}. "
            f"Check model output and class_labels in inference_config.json."
        )

    # 8. Pick predicted class
    predicted_class = _pick_class(proba_row, class_labels, thresholds)

    p_low, p_medium, p_high = float(proba_row[0]), float(proba_row[1]), float(proba_row[2])

    logger.info(
        "[%s] Inference — predicted=%s p_low=%.3f p_medium=%.3f p_high=%.3f bundle=%s",
        asset_id, predicted_class.value, p_low, p_medium, p_high, manifest.bundle_version,
    )

    return PredictionOutput(
        predicted_class=predicted_class,
        p_low=p_low,
        p_medium=p_medium,
        p_high=p_high,
        bundle_version=manifest.bundle_version,
    )
