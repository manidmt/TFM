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

import io
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class _CPUUnpickler(pickle.Unpickler):
    """Unpickler that forces torch tensors to load on CPU.

    Needed when a bundle was serialized on a CUDA machine but is being
    deserialized on a CPU-only host (e.g. Raspberry Pi 5).  Without this,
    torch raises ``RuntimeError: Attempting to deserialize object on a
    CUDA device but torch.cuda.is_available() is False``.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch  # imported lazily to avoid a hard dep for xgb-only bundles
            return lambda b: torch.load(
                io.BytesIO(b), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


def _force_model_to_cpu(model: Any) -> None:
    """Force a loaded model onto CPU after unpickling on a CPU-only host.

    Required for TabPFN models trained on CUDA: their internal
    ``executor_`` carries a per-device model cache keyed by ``cuda:0``,
    which causes ``torch.as_tensor(..., device=cuda:0)`` calls during
    ``predict_proba`` to crash with
    ``AssertionError: Torch not compiled with CUDA enabled``.

    Strategy:
    1.  If the object is a ``TabPFNClassifier`` / ``TabPFNRegressor``,
        call its own ``.to("cpu")`` which correctly invokes
        ``executor_.to([cpu_device], ...)`` and rebuilds the internal
        model cache keyed by the CPU device.
    2.  For any other object (or as a safety net), walk the instance
        graph and rewrite cached ``device`` / ``device_`` attributes to
        CPU.  The walk is bounded in depth and tracks visited ids.
    """
    try:
        import torch  # imported lazily
    except ImportError:
        return

    cpu = torch.device("cpu")

    # --- 1. Preferred path: use TabPFN's own relocation API. -----------
    try:
        from tabpfn.classifier import TabPFNClassifier  # type: ignore[import]
    except ImportError:
        TabPFNClassifier = None  # type: ignore[assignment]
    try:
        from tabpfn.regressor import TabPFNRegressor  # type: ignore[import]
    except ImportError:
        TabPFNRegressor = None  # type: ignore[assignment]

    tabpfn_types = tuple(t for t in (TabPFNClassifier, TabPFNRegressor) if t is not None)
    if tabpfn_types and isinstance(model, tabpfn_types):
        try:
            model.to("cpu")
            logger.info(
                "Relocated TabPFN estimator to CPU via estimator.to('cpu')."
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TabPFN .to('cpu') failed (%s); falling back to attribute walk.",
                exc,
            )

    # --- 2. Fallback: best-effort attribute walker. -------------------
    visited: set[int] = set()

    def _walk(obj: Any, depth: int = 0) -> None:
        if depth > 6 or id(obj) in visited:
            return
        visited.add(id(obj))

        if isinstance(obj, torch.nn.Module):
            try:
                obj.to(cpu)
            except Exception:  # noqa: BLE001
                pass

        if not hasattr(obj, "__dict__"):
            return

        for name, value in list(vars(obj).items()):
            if name in ("device", "device_"):
                if isinstance(value, torch.device) or (
                    isinstance(value, str) and "cuda" in value.lower()
                ):
                    try:
                        setattr(obj, name, cpu)
                    except Exception:  # noqa: BLE001
                        pass
                continue
            if value is None or isinstance(value, (int, float, str, bool, bytes)):
                continue
            if isinstance(value, (list, tuple)):
                for item in value:
                    _walk(item, depth + 1)
                continue
            if isinstance(value, dict):
                for item in value.values():
                    _walk(item, depth + 1)
                continue
            if hasattr(value, "__dict__") or isinstance(value, torch.nn.Module):
                _walk(value, depth + 1)

    _walk(model)


def _patch_tabpfn_memory_check() -> None:
    """Bypass TabPFN's CUDA memory planner on CPU-only hosts.

    TabPFN calls ``should_save_peak_mem`` at inference time, which in turn
    invokes ``torch.cuda.mem_get_info`` for every device in its cached
    device list.  On a CPU-only torch build this raises
    ``AssertionError: Torch not compiled with CUDA enabled``.  When CUDA
    is not available we replace the function with a constant ``False``,
    which disables the peak-memory safeguard (harmless for single-sample
    inference on a daily batch).
    """
    try:
        import torch

        if torch.cuda.is_available():
            return  # CUDA present → keep original behaviour
    except ImportError:
        return

    try:
        import tabpfn.architectures.base.memory as _tabpfn_memory
    except ImportError:
        return  # TabPFN not installed; nothing to patch

    if getattr(_tabpfn_memory, "_qr_cpu_patched", False):
        return

    _noop = lambda *args, **kwargs: False  # noqa: E731

    # Patch at the definition site.
    _tabpfn_memory.should_save_peak_mem = _noop  # type: ignore[assignment]
    _tabpfn_memory._qr_cpu_patched = True  # type: ignore[attr-defined]

    # Patch at every *use* site.  Modules that did
    # ``from tabpfn.architectures.base.memory import should_save_peak_mem``
    # hold their own local binding which is not updated by the line above.
    # tabpfn.inference is the known consumer; patch defensively in case of
    # future re-exports.
    for mod_name in ("tabpfn.inference",):
        try:
            import importlib
            _mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(_mod, "should_save_peak_mem"):
            setattr(_mod, "should_save_peak_mem", _noop)

    logger.info(
        "Patched should_save_peak_mem in tabpfn.architectures.base.memory "
        "and tabpfn.inference to return False on CPU-only host."
    )

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
    # _CPUUnpickler transparently maps CUDA tensors to CPU so bundles
    # trained on GPU can be loaded on the CPU-only RPi5.
    # _force_model_to_cpu then moves TabPFN's executor to CPU via its own
    # API; after that, TabPFN's automatic CPU memory-saving heuristic
    # kicks in and peak RSS stays within the RPi5's 8GB budget.
    pkl = model_dir / "model.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            model = _CPUUnpickler(f).load()
        _force_model_to_cpu(model)
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
            calibrator = _CPUUnpickler(f).load()
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
