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

    horizon = cfg.horizon
    output_df = full_df.sort_values(["ticker", "date"]).copy()

    forecast_mean_col = f"sarimax_fcst_mean_h{horizon}"
    forecast_se_col = f"sarimax_fcst_se_h{horizon}"
    forecast_ci_width_col = f"sarimax_ci_width_h{horizon}"

    output_df[forecast_mean_col] = np.nan
    output_df[forecast_se_col] = np.nan
    output_df[forecast_ci_width_col] = np.nan
    output_df["sarimax_resid"] = np.nan
    output_df["sarimax_resid_std"] = np.nan

    for ticker in output_df["ticker"].unique():
        fitted_result = fitted_models.get(ticker)
        if fitted_result is None:
            continue

        ticker_idx = output_df.index[output_df["ticker"] == ticker]
        ticker_df = output_df.loc[ticker_idx].sort_values("date").copy()

        target_values = ticker_df[cfg.target_col].to_numpy(dtype=float)
        if cfg.log_transform:
            target_values = _safe_log(target_values)

        exog_values = None
        exog_dim = 0
        if cfg.exog_cols:
            exog_values = ticker_df[list(cfg.exog_cols)].to_numpy(dtype=float)
            exog_dim = exog_values.shape[1]

        fitted_params = getattr(fitted_result, "params", None)
        if fitted_params is None:
            continue

        min_obs_for_state = max(10, cfg.order[0] + cfg.order[2] + cfg.order[1] + 1)
        if len(ticker_df) <= min_obs_for_state + horizon:
            continue

        initial_model = SARIMAX(
            target_values[:min_obs_for_state],
            exog=exog_values[:min_obs_for_state] if exog_values is not None else None,
            order=cfg.order,
            seasonal_order=cfg.seasonal_order,
            trend=cfg.trend,
            enforce_stationarity=cfg.enforce_stationarity,
            enforce_invertibility=cfg.enforce_invertibility,
        )
        state_result = initial_model.filter(fitted_params)

        forecast_mean = np.full(len(ticker_df), np.nan, dtype=float)
        forecast_se = np.full(len(ticker_df), np.nan, dtype=float)
        forecast_ci_width = np.full(len(ticker_df), np.nan, dtype=float)
        residuals = np.full(len(ticker_df), np.nan, dtype=float)
        residuals_std = np.full(len(ticker_df), np.nan, dtype=float)

        for idx_i in range(min_obs_for_state - 1, len(ticker_df) - horizon):
            try:
                if getattr(state_result, "resid", None) is not None and len(state_result.resid) > 0:
                    residual_value = float(state_result.resid[-1])
                    residuals[idx_i] = residual_value
                    scale_value = float(getattr(state_result, "scale", np.nan))
                    if np.isfinite(scale_value) and scale_value > 0:
                        residuals_std[idx_i] = residual_value / np.sqrt(scale_value)

                if exog_values is not None:
                    last_exog_row = exog_values[idx_i].reshape(1, exog_dim)
                    exog_future_values = np.repeat(last_exog_row, repeats=horizon, axis=0)
                    forecast_result = state_result.get_forecast(steps=horizon, exog=exog_future_values)
                else:
                    forecast_result = state_result.get_forecast(steps=horizon)

                predicted_mean = forecast_result.predicted_mean
                mean_at_horizon = float(predicted_mean[-1])

                ci = forecast_result.conf_int(alpha=0.05)
                if hasattr(ci, "iloc"):
                    ci_low, ci_high = float(ci.iloc[-1, 0]), float(ci.iloc[-1, 1])
                else:
                    ci_low, ci_high = float(ci[-1, 0]), float(ci[-1, 1])

                se_at_horizon = _forecast_se_from_ci(ci_low, ci_high, alpha=0.05)
                forecast_mean[idx_i] = mean_at_horizon
                forecast_se[idx_i] = se_at_horizon
                forecast_ci_width[idx_i] = ci_high - ci_low

            except Exception:
                pass

            try:
                next_target = np.array([target_values[idx_i + 1]], dtype=float)
                if exog_values is not None:
                    next_exog = exog_values[idx_i + 1].reshape(1, exog_dim)
                    state_result = state_result.append(endog=next_target, exog=next_exog, refit=False)
                else:
                    state_result = state_result.append(endog=next_target, refit=False)
            except Exception:
                window_start = max(0, idx_i + 1 - min_obs_for_state)
                fallback_model = SARIMAX(
                    target_values[window_start : idx_i + 2],
                    exog=exog_values[window_start : idx_i + 2] if exog_values is not None else None,
                    order=cfg.order,
                    seasonal_order=cfg.seasonal_order,
                    trend=cfg.trend,
                    enforce_stationarity=cfg.enforce_stationarity,
                    enforce_invertibility=cfg.enforce_invertibility,
                )
                state_result = fallback_model.filter(fitted_params)

        if cfg.log_transform:
            forecast_mean = np.exp(forecast_mean)

        ticker_df[forecast_mean_col] = forecast_mean
        ticker_df[forecast_se_col] = forecast_se
        ticker_df[forecast_ci_width_col] = forecast_ci_width
        ticker_df["sarimax_resid"] = residuals
        ticker_df["sarimax_resid_std"] = residuals_std

        output_df.loc[
            ticker_df.index,
            [
                forecast_mean_col,
                forecast_se_col,
                forecast_ci_width_col,
                "sarimax_resid",
                "sarimax_resid_std",
            ],
        ] = ticker_df[
            [
                forecast_mean_col,
                forecast_se_col,
                forecast_ci_width_col,
                "sarimax_resid",
                "sarimax_resid_std",
            ]
        ]

    return output_df