'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-07

@description: HAR-RV forecasting features for hybrid regime classification.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HarRvConfig:
    horizon: int = 5
    target_col: str = "rv_20"
    lag_1: int = 1
    lag_week: int = 5
    lag_month: int = 22
    exog_cols: tuple[str, ...] = ()
    train_nobs_by_ticker: dict[str, int] | None = None


def _lags_ok(cfg: HarRvConfig) -> None:
    if cfg.lag_1 <= 0 or cfg.lag_week <= 0 or cfg.lag_month <= 0:
        raise ValueError("HAR lags must be positive integers.")
    if cfg.horizon <= 0:
        raise ValueError("horizon must be >= 1")


def _row_from_history(
    history: np.ndarray,
    *,
    lag_1: int,
    lag_week: int,
    lag_month: int,
    exog_row: np.ndarray | None,
) -> np.ndarray:
    x = [
        1.0,
        float(history[-lag_1]),
        float(np.mean(history[-lag_week:])),
        float(np.mean(history[-lag_month:])),
    ]
    if exog_row is not None:
        x.extend([float(v) for v in exog_row])
    return np.asarray(x, dtype=float)


def fit_har_rv(train_df: pd.DataFrame, cfg: HarRvConfig) -> dict[str, dict[str, Any] | None]:
    _lags_ok(cfg)
    if cfg.target_col not in train_df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in train_df")

    fitted: dict[str, dict[str, Any] | None] = {}
    min_hist = int(max(cfg.lag_1, cfg.lag_week, cfg.lag_month))

    for ticker in train_df["ticker"].unique():
        td = train_df[train_df["ticker"] == ticker].sort_values("date").copy()
        if cfg.train_nobs_by_ticker is not None:
            expected = cfg.train_nobs_by_ticker.get(str(ticker))
            if expected is not None and expected != len(td):
                print(
                    f"[har] Warning: ticker={ticker} train rows mismatch "
                    f"(expected={expected}, got={len(td)})."
                )

        y = td[cfg.target_col].to_numpy(dtype=float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        exog = None
        if cfg.exog_cols:
            missing = set(cfg.exog_cols) - set(td.columns)
            if missing:
                raise ValueError(f"HAR exog columns {missing} not found for ticker={ticker}")
            exog = td[list(cfg.exog_cols)].to_numpy(dtype=float)
            exog = np.nan_to_num(exog, nan=0.0, posinf=0.0, neginf=0.0)

        x_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        for i in range(min_hist, len(y)):
            hist = y[:i]
            row = _row_from_history(
                hist,
                lag_1=cfg.lag_1,
                lag_week=cfg.lag_week,
                lag_month=cfg.lag_month,
                exog_row=exog[i] if exog is not None else None,
            )
            if not np.isfinite(row).all() or not np.isfinite(y[i]):
                continue
            x_rows.append(row)
            y_rows.append(float(y[i]))

        if len(x_rows) < 20:
            print(f"[har] Warning: insufficient rows for ticker={ticker} (n={len(x_rows)})")
            fitted[ticker] = None
            continue

        x_mat = np.vstack(x_rows)
        y_vec = np.asarray(y_rows, dtype=float)
        beta, *_ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
        y_hat = x_mat @ beta
        resid = y_vec - y_hat
        n_obs = int(resid.shape[0])
        k_params = int(x_mat.shape[1])  # includes intercept
        denom = max(n_obs - k_params, 1)
        resid_std = float(np.sqrt(np.sum(resid ** 2) / denom))

        fitted[ticker] = {
            "beta": beta.astype(float),
            "resid_std": resid_std,
            "min_hist": int(min_hist),
            "exog_dim": int(exog.shape[1] if exog is not None else 0),
            "nobs_train": int(len(td)),
        }

    return fitted


def make_har_rv_features(
    full_df: pd.DataFrame,
    fitted: dict[str, dict[str, Any] | None],
    cfg: HarRvConfig,
) -> pd.DataFrame:
    _lags_ok(cfg)
    if cfg.target_col not in full_df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in full_df")

    h = int(cfg.horizon)
    out = full_df.sort_values(["ticker", "date"]).copy()
    mean_col = f"har_fcst_mean_h{h}"
    se_col = f"har_fcst_se_h{h}"
    ci_col = f"har_ci_width_h{h}"
    out[mean_col] = np.nan
    out[se_col] = np.nan
    out[ci_col] = np.nan
    out["har_resid"] = np.nan
    out["har_resid_std"] = np.nan

    for ticker in out["ticker"].unique():
        tk_fit = fitted.get(ticker)
        if not tk_fit:
            continue

        idx = out.index[out["ticker"] == ticker]
        sub = out.loc[idx].sort_values("date").copy()

        y = sub[cfg.target_col].to_numpy(dtype=float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        exog = None
        if cfg.exog_cols:
            missing = set(cfg.exog_cols) - set(sub.columns)
            if missing:
                raise ValueError(
                    f"HAR exog columns {missing} not found in full_df for ticker={ticker}"
                )
            exog = sub[list(cfg.exog_cols)].to_numpy(dtype=float)
            exog = np.nan_to_num(exog, nan=0.0, posinf=0.0, neginf=0.0)

        beta = np.asarray(tk_fit["beta"], dtype=float)
        resid_std_train = float(tk_fit.get("resid_std", np.nan))
        min_hist = int(tk_fit.get("min_hist", max(cfg.lag_1, cfg.lag_week, cfg.lag_month)))

        n = len(sub)
        pred_h = np.full(n, np.nan, dtype=float)
        pred_se = np.full(n, np.nan, dtype=float)
        pred_ci = np.full(n, np.nan, dtype=float)
        resid = np.full(n, np.nan, dtype=float)
        resid_std = np.full(n, np.nan, dtype=float)

        for i in range(min_hist, n - h):
            hist_obs = y[:i]
            x_now = _row_from_history(
                hist_obs,
                lag_1=cfg.lag_1,
                lag_week=cfg.lag_week,
                lag_month=cfg.lag_month,
                exog_row=exog[i] if exog is not None else None,
            )
            y_hat_now = float(x_now @ beta)
            resid_val = float(y[i] - y_hat_now)
            resid[i] = resid_val
            if np.isfinite(resid_std_train) and resid_std_train > 0:
                resid_std[i] = resid_val / resid_std_train

            hist = list(y[: i + 1])
            for step in range(1, h + 1):
                if exog is not None:
                    # Carry-forward current exog to avoid look-ahead leakage.
                    ex_row = exog[i]
                else:
                    ex_row = None
                x_future = _row_from_history(
                    np.asarray(hist, dtype=float),
                    lag_1=cfg.lag_1,
                    lag_week=cfg.lag_week,
                    lag_month=cfg.lag_month,
                    exog_row=ex_row,
                )
                hist.append(float(x_future @ beta))

            pred = float(hist[-1])
            pred_h[i] = pred
            if np.isfinite(resid_std_train) and resid_std_train > 0:
                se = float(resid_std_train * np.sqrt(h))
                pred_se[i] = se
                pred_ci[i] = float(2.0 * 1.959963984540054 * se)

        sub[mean_col] = pred_h
        sub[se_col] = pred_se
        sub[ci_col] = pred_ci
        sub["har_resid"] = resid
        sub["har_resid_std"] = resid_std
        out.loc[sub.index, [mean_col, se_col, ci_col, "har_resid", "har_resid_std"]] = sub[
            [mean_col, se_col, ci_col, "har_resid", "har_resid_std"]
        ]

    return out
