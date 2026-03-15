'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-12

@description: GKG-specialized change/no-change detector for regime transitions.
'''

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import set_global_seed, to_numpy_x, to_numpy_y


_GKG_PREFIX_RE = re.compile(
    r"^(news_count|tone_mean|tone_std|tone_neg_share|log_news|log_news_count|"
    r"attn_z|attention_z|attention_shock_z|attn_shock|tone_change_1d|tone_change_3d|"
    r"neg_share_change_1d|topic_share|topic_share_z)(?:_.+)?$"
)
_GKG_SHOCK_RE = re.compile(r"^shock_days_w\d+(?:_.+)?$")
_GKG_TONE_WINDOW_RE = re.compile(r"^tone_(mean|std)_w\d+(?:_.+)?$")
_GKG_ROLL_RE = re.compile(
    r"^(news_count|tone_mean|tone_std|tone_neg_share|log_news|log_news_count|attn_z|"
    r"attention_z|attention_shock_z|attn_shock|tone_change_1d|tone_change_3d|"
    r"neg_share_change_1d|topic_share|topic_share_z)_(sum|mean|std)_w\d+(?:_.+)?$"
)


@dataclass(frozen=True)
class GkgChangeDetectorConfig:
    model_type: str = "logit"  # logit | xgb | tabpfn
    calibration_method: str = "none"  # none | platt | isotonic
    random_state: int = 42
    seed: int = 42
    # XGBoost params (used only when model_type=xgb)
    xgb_n_estimators: int = 250
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_min_child_weight: float = 1.0
    xgb_reg_lambda: float = 1.0
    xgb_n_jobs: int = 1
    # TabPFN params (used only when model_type=tabpfn)
    tabpfn_n_estimators: int = 8
    tabpfn_softmax_temperature: float = 0.9
    tabpfn_balance_probabilities: bool = False


@dataclass(frozen=True)
class GkgChangeDetectorArtifacts:
    model: Any
    model_type: str
    feature_names: tuple[str, ...]
    calibrator: dict[str, Any]


def _is_gkg_feature(col: str) -> bool:
    c = str(col)
    if "_x_" in c:
        return False
    return bool(
        _GKG_PREFIX_RE.match(c)
        or _GKG_SHOCK_RE.match(c)
        or _GKG_TONE_WINDOW_RE.match(c)
        or _GKG_ROLL_RE.match(c)
    )


def select_gkg_feature_columns(feature_cols: Sequence[str]) -> list[str]:
    """Return only GKG-derived feature columns from a model feature space."""
    out: list[str] = []
    for c in feature_cols:
        if _is_gkg_feature(str(c)):
            out.append(str(c))
    return out


def make_model(cfg: GkgChangeDetectorConfig):
    model_type = str(cfg.model_type).lower().strip()
    if model_type == "logit":
        set_global_seed(cfg.seed, use_torch=False)
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=2500,
                        class_weight="balanced",
                        random_state=int(cfg.random_state),
                    ),
                ),
            ]
        )
    if model_type == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "xgboost no está instalado. Instala con: poetry add xgboost"
            ) from e
        set_global_seed(cfg.seed, use_torch=False)
        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=int(cfg.xgb_n_estimators),
            max_depth=int(cfg.xgb_max_depth),
            learning_rate=float(cfg.xgb_learning_rate),
            subsample=float(cfg.xgb_subsample),
            colsample_bytree=float(cfg.xgb_colsample_bytree),
            min_child_weight=float(cfg.xgb_min_child_weight),
            reg_lambda=float(cfg.xgb_reg_lambda),
            n_jobs=int(cfg.xgb_n_jobs),
            eval_metric="logloss",
            random_state=int(cfg.random_state),
        )
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
    raise ValueError(f"Invalid GKG change detector model_type={cfg.model_type!r}")


def _fit_binary_calibrator(
    p_change: np.ndarray,
    y_true: np.ndarray,
    method: str,
) -> dict[str, Any]:
    method_l = str(method).lower().strip()
    p = np.asarray(p_change, dtype=float).reshape(-1)
    y = np.asarray(y_true, dtype=int).reshape(-1)
    if p.shape[0] != y.shape[0]:
        raise ValueError(
            f"Calibration inputs have different lengths: proba={p.shape[0]} y={y.shape[0]}"
        )
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)

    if method_l in {"", "none"}:
        return {"method": "none", "model": None}
    if int(np.unique(y).size) < 2:
        return {"method": "none", "model": None}
    if method_l == "platt":
        clf = LogisticRegression(max_iter=300, solver="lbfgs", random_state=42)
        clf.fit(p.reshape(-1, 1), y)
        return {"method": "platt", "model": clf}
    if method_l == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p, y)
        return {"method": "isotonic", "model": ir}
    raise ValueError(
        f"Invalid calibration method={method!r}. Use one of: none, platt, isotonic."
    )


def _apply_binary_calibrator(
    p_change: np.ndarray,
    calibrator: dict[str, Any] | None,
) -> np.ndarray:
    p = np.asarray(p_change, dtype=float).reshape(-1)
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)
    if not calibrator or str(calibrator.get("method", "none")) == "none":
        return p
    method = str(calibrator.get("method", "none"))
    model = calibrator.get("model")
    if model is None:
        return p
    if method == "platt":
        out = np.asarray(model.predict_proba(p.reshape(-1, 1))[:, 1], dtype=float)
    elif method == "isotonic":
        out = np.asarray(model.predict(p), dtype=float)
    else:
        out = p
    out = np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0)


def fit(
    cfg: GkgChangeDetectorConfig,
    X_train,
    y_change_train,
    *,
    feature_names: Sequence[str],
) -> GkgChangeDetectorArtifacts:
    x = to_numpy_x(X_train, dtype=np.float32)
    y = to_numpy_y(y_change_train)
    if x.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape={x.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"X_train and y_change_train length mismatch: {x.shape[0]} vs {y.shape[0]}"
        )
    if x.shape[0] == 0:
        raise ValueError("Cannot fit change detector with 0 rows.")
    if int(np.unique(y).size) < 2:
        raise ValueError("Change detector training labels require both classes (0 and 1).")

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
    return GkgChangeDetectorArtifacts(
        model=model,
        model_type=str(cfg.model_type).lower().strip(),
        feature_names=tuple(str(c) for c in feature_names),
        calibrator=calibrator,
    )


def predict_proba_change(model: Any, X) -> np.ndarray:
    x = to_numpy_x(X, dtype=np.float32)
    p = np.asarray(model.predict_proba(x), dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError(f"Expected predict_proba with >=2 columns, got shape={p.shape}")
    # Handle either [P(0), P(1)] or model-class-ordered matrices robustly.
    if p.shape[1] == 2:
        out = p[:, 1]
    else:
        out = p[:, -1]
    out = np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(out.astype(float), 0.0, 1.0)


def predict_proba(
    artifacts: GkgChangeDetectorArtifacts,
    X,
) -> np.ndarray:
    """Return calibrated [p_no_change, p_change] probabilities."""
    p_change_raw = predict_proba_change(artifacts.model, X)
    p_change = _apply_binary_calibrator(p_change_raw, artifacts.calibrator)
    p_no_change = 1.0 - p_change
    out = np.column_stack([p_no_change, p_change]).astype(float)
    out = np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
    out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
    out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
    return out


def build_design_matrix(
    *,
    X_all: np.ndarray,
    feature_cols: Sequence[str],
    gkg_feature_cols: Sequence[str] | None = None,
    context_cols: Iterable[str] = (),
    regime_prev: np.ndarray | None = None,
    sigma_t: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a design matrix using only GKG features plus optional minimal context."""
    feature_cols_l = [str(c) for c in feature_cols]
    selected = (
        [str(c) for c in gkg_feature_cols]
        if gkg_feature_cols is not None
        else select_gkg_feature_columns(feature_cols_l)
    )
    idx_map = {c: i for i, c in enumerate(feature_cols_l)}
    use_idx = [idx_map[c] for c in selected if c in idx_map]

    base = np.asarray(X_all, dtype=np.float32)
    mats: list[np.ndarray] = []
    names: list[str] = []
    if use_idx:
        mats.append(base[:, use_idx].astype(np.float32))
        names.extend([feature_cols_l[i] for i in use_idx])

    context_norm = [str(c).strip().lower() for c in context_cols if str(c).strip()]
    if "regime_prev" in context_norm:
        if regime_prev is None:
            raise ValueError("context 'regime_prev' requested but regime_prev is None.")
        rp = np.asarray(regime_prev, dtype=float).reshape(-1, 1)
        if rp.shape[0] != base.shape[0]:
            raise ValueError("regime_prev length must match X_all rows.")
        mats.append(rp.astype(np.float32))
        names.append("ctx_regime_prev")
    if "sigma_t" in context_norm:
        if sigma_t is None:
            raise ValueError("context 'sigma_t' requested but sigma_t is None.")
        sg = np.asarray(sigma_t, dtype=float).reshape(-1, 1)
        if sg.shape[0] != base.shape[0]:
            raise ValueError("sigma_t length must match X_all rows.")
        mats.append(sg.astype(np.float32))
        names.append("ctx_sigma_t")

    if not mats:
        return np.empty((base.shape[0], 0), dtype=np.float32), []
    return np.concatenate(mats, axis=1).astype(np.float32), names
