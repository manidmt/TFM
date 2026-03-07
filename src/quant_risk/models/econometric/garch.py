'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-12

@description: GARCH-based volatility forecasting features for hybrid regime classification.

Design choices:
- Fit parameters ONLY on TRAIN (per ticker).
- Build features on full_df with walk-forward expanding windows up to t only.
- Forecast at row t corresponds to volatility for t+h (no look-ahead).

TODO(optimization): current recursive feature generation prioritizes correctness over speed.
It rebuilds a fixed-parameter model on expanding windows and can be optimized with a
stateful recursion if needed for very long histories.
'''

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
from arch.univariate import ConstantMean, GARCH, Normal, SkewStudent, StudentsT, ZeroMean

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GarchConfig:
    p: int = 1
    q: int = 1
    dist: str = "normal"
    mean: str = "zero"
    vol: str = "Garch"
    horizon: int = 5
    target_col: str = "logret"
    agg: str = "last"
    annualize: bool = False
    scale: float = 100.0
    train_nobs_by_ticker: dict[str, int] | None = None


def _build_mean_model(y: np.ndarray, cfg: GarchConfig):
    mean_kind = cfg.mean.lower()
    if mean_kind == "zero":
        model = ZeroMean(y)
    elif mean_kind in {"constant", "const"}:
        model = ConstantMean(y)
    else:
        raise ValueError(f"Unsupported GARCH mean='{cfg.mean}'. Use 'zero' or 'constant'.")

    if cfg.vol.lower() != "garch":
        raise ValueError(f"Unsupported volatility process vol='{cfg.vol}'. Only 'Garch' is supported.")

    model.volatility = GARCH(p=cfg.p, o=0, q=cfg.q)

    dist_kind = cfg.dist.lower()
    if dist_kind in {"normal", "gaussian"}:
        model.distribution = Normal()
    elif dist_kind in {"t", "studentt", "student-t", "tstudent", "students_t"}:
        model.distribution = StudentsT()
    elif dist_kind in {"skewt", "skew_t", "skew-t", "skewstudent", "skew_student"}:
        model.distribution = SkewStudent()
    else:
        raise ValueError(
            f"Unsupported distribution dist='{cfg.dist}'. "
            "Use 'normal', 'tstudent' or 'skewt'."
        )
    return model


def fit_garch(train_df: pd.DataFrame, cfg: GarchConfig) -> dict[str, dict[str, Any] | None]:
    if cfg.target_col not in train_df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in train_df")

    fitted: dict[str, dict[str, Any] | None] = {}

    for ticker in train_df["ticker"].unique():
        td = train_df[train_df["ticker"] == ticker].sort_values("date").copy()

        if cfg.train_nobs_by_ticker is not None:
            expected = cfg.train_nobs_by_ticker.get(str(ticker))
            if expected is not None and expected != len(td):
                print(
                    f"[garch] Warning: ticker={ticker} train rows mismatch "
                    f"(expected={expected}, got={len(td)})."
                )

        y = td[cfg.target_col].to_numpy(dtype=float)
        if not np.isfinite(y).all():
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        y_scaled = y * float(cfg.scale)

        try:
            model = _build_mean_model(y_scaled, cfg)
            result = model.fit(disp="off", show_warning=False)
            fitted[ticker] = {
                "params": result.params.copy(),
                "scale": float(cfg.scale),
                "nobs_train": int(len(td)),
            }
        except Exception as e:
            print(f"[garch] Warning: fit failed for {ticker}: {e}")
            fitted[ticker] = None

    return fitted


def make_garch_features(
    full_df: pd.DataFrame,
    fitted: dict[str, dict[str, Any] | None],
    cfg: GarchConfig,
) -> pd.DataFrame:
    if cfg.target_col not in full_df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in full_df")

    h = int(cfg.horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    agg_kind = str(cfg.agg).lower()
    if agg_kind not in {"last", "mean", "rms"}:
        raise ValueError("cfg.agg must be one of: 'last', 'mean', 'rms'")

    output_df = full_df.sort_values(["ticker", "date"]).copy()
    sigma_fwd_col = f"garch_sigma_fwd_h{h}"
    var_fwd_col = f"garch_var_fwd_h{h}"
    var_mean_col = f"garch_var_mean_h{h}"
    sigma_rms_col = f"garch_sigma_rms_h{h}"

    output_df["garch_sigma_t"] = np.nan
    output_df[sigma_fwd_col] = np.nan
    output_df[var_fwd_col] = np.nan
    if agg_kind != "last":
        output_df[var_mean_col] = np.nan
        output_df[sigma_rms_col] = np.nan
    output_df["garch_resid"] = np.nan
    output_df["garch_z"] = np.nan

    ann_sigma_factor = float(np.sqrt(252.0)) if cfg.annualize else 1.0
    ann_var_factor = 252.0 if cfg.annualize else 1.0

    min_obs_for_state = max(20, cfg.p + cfg.q + 5)

    for ticker in output_df["ticker"].unique():
        ticker_fit = fitted.get(ticker)
        if not ticker_fit or "params" not in ticker_fit:
            continue

        ticker_idx = output_df.index[output_df["ticker"] == ticker]
        ticker_df = output_df.loc[ticker_idx].sort_values("date").copy()
        n = len(ticker_df)
        if n <= min_obs_for_state + h:
            continue

        y = ticker_df[cfg.target_col].to_numpy(dtype=float)
        if not np.isfinite(y).all():
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        scale = float(cfg.scale)
        y_scaled = y * scale

        sigma_t = np.full(n, np.nan, dtype=float)
        sigma_fwd = np.full(n, np.nan, dtype=float)
        var_fwd = np.full(n, np.nan, dtype=float)
        var_mean = np.full(n, np.nan, dtype=float)
        sigma_rms = np.full(n, np.nan, dtype=float)
        resid = np.full(n, np.nan, dtype=float)
        z = np.full(n, np.nan, dtype=float)
        err = 0
        max_err = max(25, int(0.10 * len(ticker_df)))

        params = ticker_fit["params"]

        for i in range(min_obs_for_state - 1, n - h):
            try:
                model_i = _build_mean_model(y_scaled[: i + 1], cfg)
                fixed_i = model_i.fix(params)

                sigma_scaled = float(fixed_i.conditional_volatility[-1])
                resid_scaled = float(fixed_i.resid[-1])

                sigma_t_val = sigma_scaled / scale
                resid_val = resid_scaled / scale

                sigma_t[i] = sigma_t_val * ann_sigma_factor
                resid[i] = resid_val

                if np.isfinite(sigma_scaled) and sigma_scaled > 0:
                    z[i] = resid_scaled / sigma_scaled

                fcast = fixed_i.forecast(horizon=h, reindex=False)
                var_path_scaled = np.asarray(fcast.variance.values[-1, :h], dtype=float)
                var_path = np.maximum(var_path_scaled / (scale * scale), 0.0)
                var_h = float(var_path[-1])
                sigma_h = float(np.sqrt(var_h))

                var_fwd[i] = var_h * ann_var_factor
                sigma_fwd[i] = sigma_h * ann_sigma_factor
                if agg_kind != "last":
                    var_mean_h = float(np.mean(var_path))
                    sigma_rms_h = float(np.sqrt(var_mean_h))
                    var_mean[i] = var_mean_h * ann_var_factor
                    sigma_rms[i] = sigma_rms_h * ann_sigma_factor
            except Exception as e:
                err += 1
                if err in (1, 5, 10) or err % 50 == 0:
                    logger.warning("[garch] ticker=%s i=%s error=%r", ticker, i, e)
                if err > max_err:
                    raise RuntimeError(
                        f"[garch] Too many errors for ticker={ticker}: {err}/{len(ticker_df)}"
                    ) from e
                continue

        ticker_df["garch_sigma_t"] = sigma_t
        ticker_df[sigma_fwd_col] = sigma_fwd
        ticker_df[var_fwd_col] = var_fwd
        if agg_kind != "last":
            ticker_df[var_mean_col] = var_mean
            ticker_df[sigma_rms_col] = sigma_rms
        ticker_df["garch_resid"] = resid
        ticker_df["garch_z"] = z

        assign_cols = ["garch_sigma_t", sigma_fwd_col, var_fwd_col, "garch_resid", "garch_z"]
        if agg_kind != "last":
            assign_cols.extend([var_mean_col, sigma_rms_col])
        output_df.loc[ticker_df.index, assign_cols] = ticker_df[assign_cols]

    return output_df
