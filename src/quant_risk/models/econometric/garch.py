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
from typing import Any

import numpy as np
import pandas as pd
from arch.univariate import ConstantMean, GARCH, Normal, ZeroMean


@dataclass(frozen=True)
class GarchConfig:
    p: int = 1
    q: int = 1
    dist: str = "normal"
    mean: str = "zero"
    vol: str = "Garch"
    horizon: int = 5
    target_col: str = "logret"
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

    if cfg.dist.lower() != "normal":
        raise ValueError(f"Unsupported distribution dist='{cfg.dist}'. Only 'normal' is supported.")

    model.distribution = Normal()
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

    output_df = full_df.sort_values(["ticker", "date"]).copy()
    sigma_fwd_col = f"garch_sigma_fwd_h{h}"
    var_fwd_col = f"garch_var_fwd_h{h}"

    output_df["garch_sigma_t"] = np.nan
    output_df[sigma_fwd_col] = np.nan
    output_df[var_fwd_col] = np.nan
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
        resid = np.full(n, np.nan, dtype=float)
        z = np.full(n, np.nan, dtype=float)

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
                var_scaled_h = float(fcast.variance.values[-1, h - 1])
                var_h = max(var_scaled_h / (scale * scale), 0.0)
                sigma_h = float(np.sqrt(var_h))

                var_fwd[i] = var_h * ann_var_factor
                sigma_fwd[i] = sigma_h * ann_sigma_factor
            except Exception:
                continue

        ticker_df["garch_sigma_t"] = sigma_t
        ticker_df[sigma_fwd_col] = sigma_fwd
        ticker_df[var_fwd_col] = var_fwd
        ticker_df["garch_resid"] = resid
        ticker_df["garch_z"] = z

        output_df.loc[
            ticker_df.index,
            ["garch_sigma_t", sigma_fwd_col, var_fwd_col, "garch_resid", "garch_z"],
        ] = ticker_df[["garch_sigma_t", sigma_fwd_col, var_fwd_col, "garch_resid", "garch_z"]]

    return output_df
