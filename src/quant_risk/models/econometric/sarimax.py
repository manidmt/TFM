'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-10

@description: SARIMAX-based volatility forecasting for hybrid regime classification.

Key design:
- Fit params ONLY on TRAIN (per ticker).
- Generate features for full_df by reusing fitted params and updating the state
  with append(refit=False) sequentially (fast, leak-minimised).
- For multi-step forecast with exog, we must provide future exog. To avoid leaking
  unknown future values, we use a simple carry-forward of last observed exog.
  (If you want strictness, set exog_cols=() for SARIMAX, or build a separate exog forecaster.)
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass(frozen=True)
class SarimaxConfig:
    order: tuple[int, int, int] = (1, 0, 1)
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    trend: Optional[str] = "c"
    horizon: int = 5
    target_col: str = "vol_fwd"
    log_transform: bool = True
    exog_cols: tuple[str, ...] = ()
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    train_nobs_by_ticker: dict[str, int] | None = None


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, 1e-12, None))


def fit_sarimax(train_df: pd.DataFrame, cfg: SarimaxConfig) -> dict[str, Any]:
    if cfg.target_col not in train_df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in train_df")

    fitted: dict[str, Any] = {}
    for ticker in train_df["ticker"].unique():
        td = train_df[train_df["ticker"] == ticker].sort_values("date").copy()

        y = td[cfg.target_col].to_numpy(dtype=float)
        if cfg.log_transform:
            y = _safe_log(y)

        exog = None
        if cfg.exog_cols:
            missing = set(cfg.exog_cols) - set(td.columns)
            if missing:
                raise ValueError(f"Exog columns {missing} not found for ticker={ticker}")
            exog = td[list(cfg.exog_cols)].to_numpy(dtype=float)

        try:
            model = SARIMAX(
                y,
                exog=exog,
                order=cfg.order,
                seasonal_order=cfg.seasonal_order,
                trend=cfg.trend,
                enforce_stationarity=cfg.enforce_stationarity,
                enforce_invertibility=cfg.enforce_invertibility,
            )
            res = model.fit(disp=False, maxiter=300)
            fitted[ticker] = res
        except Exception as e:
            print(f"[sarimax] Warning: fit failed for {ticker}: {e}")
            fitted[ticker] = None

    return fitted


def _forecast_se_from_ci(ci_low: float, ci_high: float, alpha: float = 0.05) -> float:
    # Approx: ci = mean ± z * se; z for 95% ~ 1.96
    z = 1.959963984540054
    width = ci_high - ci_low
    return float(width / (2.0 * z)) if np.isfinite(width) else float("nan")


def make_sarimax_features(
    full_df: pd.DataFrame,
    fitted_models: dict[str, Any],
    cfg: SarimaxConfig,
) -> pd.DataFrame:
    if cfg.target_col not in full_df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in full_df")

    h = cfg.horizon
    out = full_df.sort_values(["ticker", "date"]).copy()

    mean_col = f"sarimax_fcst_mean_h{h}"
    se_col = f"sarimax_fcst_se_h{h}"
    ciw_col = f"sarimax_ci_width_h{h}"

    out[mean_col] = np.nan
    out[se_col] = np.nan
    out[ciw_col] = np.nan
    out["sarimax_resid"] = np.nan
    out["sarimax_resid_std"] = np.nan

    for ticker in out["ticker"].unique():
        res0 = fitted_models.get(ticker)
        if res0 is None:
            continue

        idx = out.index[out["ticker"] == ticker]
        td = out.loc[idx].sort_values("date").copy()

        y_full = td[cfg.target_col].to_numpy(dtype=float)
        if cfg.log_transform:
            y_full = _safe_log(y_full)

        exog_full = None
        if cfg.exog_cols:
            exog_full = td[list(cfg.exog_cols)].to_numpy(dtype=float)

        params = getattr(res0, "params", None)
        if params is None:
            continue

        min_obs = max(10, cfg.order[0] + cfg.order[2] + cfg.order[1] + 1)

        for i in range(min_obs, len(td)):
            if i + h >= len(td):
                continue

            y_i = y_full[: i + 1]
            exog_i = exog_full[: i + 1] if exog_full is not None else None

            try:
                model_i = SARIMAX(
                    y_i,
                    exog=exog_i,
                    order=cfg.order,
                    seasonal_order=cfg.seasonal_order,
                    trend=cfg.trend,
                    enforce_stationarity=cfg.enforce_stationarity,
                    enforce_invertibility=cfg.enforce_invertibility,
                )
                res_i = model_i.filter(params)

                if getattr(res_i, "resid", None) is not None and len(res_i.resid) > 0:
                    resid_i = float(res_i.resid[-1])
                    td.loc[td.index[i], "sarimax_resid"] = resid_i
                    scale = float(getattr(res_i, "scale", np.nan))
                    if np.isfinite(scale) and scale > 0:
                        td.loc[td.index[i], "sarimax_resid_std"] = resid_i / np.sqrt(scale)

                if exog_full is not None:
                    last_x = exog_full[i].reshape(1, -1)
                    exog_future = np.repeat(last_x, repeats=h, axis=0)
                    fc = res_i.get_forecast(steps=h, exog=exog_future)
                else:
                    fc = res_i.get_forecast(steps=h)

                pm = fc.predicted_mean
                mean_h = float(pm[-1])

                ci = fc.conf_int(alpha=0.05)
                if hasattr(ci, "iloc"):
                    lo, hi = float(ci.iloc[-1, 0]), float(ci.iloc[-1, 1])
                else:
                    lo, hi = float(ci[-1, 0]), float(ci[-1, 1])

                se_h = _forecast_se_from_ci(lo, hi, alpha=0.05)
                ci_w = hi - lo

                td.loc[td.index[i], mean_col] = mean_h
                td.loc[td.index[i], se_col] = se_h
                td.loc[td.index[i], ciw_col] = ci_w

            except Exception:
                continue

        if cfg.log_transform:
            td[mean_col] = np.exp(td[mean_col].to_numpy(dtype=float))

        out.loc[td.index, [mean_col, se_col, ciw_col, "sarimax_resid", "sarimax_resid_std"]] = td[
            [mean_col, se_col, ciw_col, "sarimax_resid", "sarimax_resid_std"]
        ]

    return out

