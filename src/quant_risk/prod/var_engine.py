'''
@author: Manuel Díaz-Meco Terrés

@description: Regime-conditioned Historical Simulation VaR engine.

Algorithm
---------
For a portfolio with positions [(source_ticker, normalized_weight)]:

1. Compute daily simple returns from raw_prices per ticker.
2. Assign each (ticker, date) a regime label — low/medium/high — using the
   same per-ticker tercile thresholds as the public regimes/history endpoint
   (approx_quantile over labels_regime.vol_fwd, horizon=5).
3. For each date, derive the *portfolio-level* regime as the dominant regime
   by weight: the regime that accounts for the largest fraction of portfolio
   weight on that date.
4. Compute the weighted portfolio return per date.
5. Filter dates to those whose portfolio regime matches regime_filter
   (or keep all dates when regime_filter == "all").
6. Return the sorted distribution plus VaR/CVaR metrics.

The regime label on date t in labels_regime describes the *forward* realized
volatility from t+1 to t+5.  Filtering same-day returns by this label is
equivalent to asking: "on days when the market was about to be [high]-vol,
what did the portfolio return?"  This is the most scenario-rich interpretation
(~420 obs per regime tier over a 5-year history) and is academically standard.
'''

from __future__ import annotations

import math
from dataclasses import dataclass

import duckdb


@dataclass
class VaRResult:
    var_95: float          # positive — e.g. 0.023 means 2.3% potential 1-day loss
    var_99: float
    cvar_95: float         # Expected Shortfall: mean of worst 5% of scenarios
    regime_filter: str     # "low" | "medium" | "high" | "all"
    scenario_count: int    # number of historical days after regime filter
    low_sample_warning: bool   # True when scenario_count < 30
    histogram: list[float]     # sorted portfolio daily returns (ascending)


def compute_portfolio_var(
    research_db: duckdb.DuckDBPyConnection,
    ticker_weights: list[tuple[str, float]],
    regime_filter: str = "all",
) -> VaRResult:
    """Compute regime-conditioned Historical Simulation VaR.

    Parameters
    ----------
    research_db:
        Open (read-only) connection to financial_data.duckdb.
    ticker_weights:
        List of (source_ticker, normalized_weight) pairs.  Weights must sum
        to 1.0.  Tickers must exist in raw_prices and labels_regime.
    regime_filter:
        One of "low", "medium", "high", "all".  Filters historical dates to
        those whose portfolio-level regime matches this value.

    Returns
    -------
    VaRResult
        var_95 / var_99 are expressed as positive fractions of portfolio value
        (e.g. 0.023 = 2.3% potential loss).  histogram is sorted ascending
        (worst returns first).
    """
    if not ticker_weights:
        raise ValueError("ticker_weights must not be empty")
    if regime_filter not in ("low", "medium", "high", "all"):
        raise ValueError(f"Invalid regime_filter: {regime_filter!r}")

    # Inline VALUES table — tickers come from the trusted asset catalog.
    values_rows = ", ".join(
        f"('{ticker}', {float(weight):.10f})"
        for ticker, weight in ticker_weights
    )
    n_assets = len(ticker_weights)

    row = research_db.execute(f"""
        WITH tw AS (
            SELECT * FROM (VALUES {values_rows}) AS t(ticker, weight)
        ),
        thresholds AS (
            SELECT lr.ticker,
                   approx_quantile(lr.vol_fwd, 0.333) AS q33,
                   approx_quantile(lr.vol_fwd, 0.667) AS q67
            FROM labels_regime lr
            JOIN tw ON lr.ticker = tw.ticker
            WHERE lr.horizon = 5
            GROUP BY lr.ticker
        ),
        asset_regime AS (
            SELECT lr.ticker, lr.date,
                   CASE WHEN lr.vol_fwd <= t.q33 THEN 'low'
                        WHEN lr.vol_fwd <= t.q67 THEN 'medium'
                        ELSE 'high' END AS regime
            FROM labels_regime lr
            JOIN thresholds t ON lr.ticker = t.ticker
            WHERE lr.horizon = 5
        ),
        daily_rets AS (
            SELECT p.ticker, p.date,
                   (p.close / LAG(p.close) OVER (
                       PARTITION BY p.ticker ORDER BY p.date
                   )) - 1.0 AS ret
            FROM raw_prices p
            JOIN tw ON p.ticker = tw.ticker
            WHERE p.close IS NOT NULL
        ),
        portfolio_daily AS (
            SELECT
                dr.date,
                SUM(dr.ret * tw.weight) AS port_ret,
                CASE
                    WHEN SUM(CASE WHEN ar.regime = 'low'    THEN tw.weight ELSE 0.0 END) >=
                         GREATEST(
                             SUM(CASE WHEN ar.regime = 'medium' THEN tw.weight ELSE 0.0 END),
                             SUM(CASE WHEN ar.regime = 'high'   THEN tw.weight ELSE 0.0 END))
                    THEN 'low'
                    WHEN SUM(CASE WHEN ar.regime = 'medium' THEN tw.weight ELSE 0.0 END) >=
                         SUM(CASE WHEN ar.regime = 'high'   THEN tw.weight ELSE 0.0 END)
                    THEN 'medium'
                    ELSE 'high'
                END AS port_regime
            FROM daily_rets dr
            JOIN tw ON dr.ticker = tw.ticker
            JOIN asset_regime ar
                ON dr.ticker = ar.ticker AND dr.date = ar.date
            WHERE dr.ret IS NOT NULL
            GROUP BY dr.date
            HAVING COUNT(DISTINCT dr.ticker) = {n_assets}
        )
        SELECT LIST(port_ret ORDER BY port_ret) AS distribution
        FROM portfolio_daily
        WHERE (? = 'all' OR port_regime = ?)
    """, [regime_filter, regime_filter]).fetchone()

    distribution: list[float] = row[0] if row and row[0] else []
    n = len(distribution)

    if n == 0:
        return VaRResult(
            var_95=0.0, var_99=0.0, cvar_95=0.0,
            regime_filter=regime_filter,
            scenario_count=0,
            low_sample_warning=True,
            histogram=[],
        )

    # Historical simulation: sort ascending, take percentile by floor index
    cutoff_95 = max(1, math.floor(0.05 * n))
    cutoff_99 = max(1, math.floor(0.01 * n))

    var_95 = -distribution[cutoff_95 - 1]
    var_99 = -distribution[cutoff_99 - 1]
    cvar_95 = -(sum(distribution[:cutoff_95]) / cutoff_95)

    return VaRResult(
        var_95=round(var_95, 6),
        var_99=round(var_99, 6),
        cvar_95=round(cvar_95, 6),
        regime_filter=regime_filter,
        scenario_count=n,
        low_sample_warning=(n < 30),
        histogram=distribution,
    )
