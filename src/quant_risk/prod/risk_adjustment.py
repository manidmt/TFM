"""Idiosyncratic risk adjustment for portfolio analysis.

Tilts proxy regime probabilities using each asset's own 2-year historical
volatility and excess kurtosis relative to its proxy source ticker.

Algorithm
---------
For each position where the label exists in raw_prices:
  1. Compute annualized σ for the asset and its proxy source ticker.
  2. vol_ratio = clamp(σ_asset / σ_proxy, 0.25, 4.0)
  3. kurt_scale = 1 + 0.15 × max(0, (κ_asset − κ_proxy) / max(|κ_proxy|, 1.0))
  4. multiplier m = clamp(vol_ratio × kurt_scale, 0.25, 4.0)
  5. Tilt: [p_low/m, p_medium, p_high×m] → renormalize
     (p_medium is unchanged — only the tails are redistributed)

Falls back gracefully (has_risk_adjustment=False) when:
  - Ticker not found in raw_prices
  - Fewer than MIN_OBS return observations in the window
  - Proxy ticker missing or has near-zero volatility
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import duckdb
import numpy as np

logger = logging.getLogger(__name__)

_MIN_OBS = 30
_CLAMP_MIN = 0.25
_CLAMP_MAX = 4.0
_KURT_WEIGHT = 0.15
_CLASS_ORDER = ("low", "medium", "high")

# Allow only characters that appear in real ticker symbols
_SAFE_TICKER_RE = re.compile(r"[^A-Za-z0-9.\^\-=]")


@dataclass
class RiskAdjustment:
    vol_asset: float          # annualized σ of the position's ticker
    vol_proxy: float          # annualized σ of the proxy source ticker
    vol_ratio: float          # σ_asset / σ_proxy, clamped to [0.25, 4.0]
    kurt_scale: float         # kurtosis tail-risk multiplier (≥ 1.0)
    multiplier: float         # combined vol_ratio × kurt_scale, clamped
    adj_p_low: float
    adj_p_medium: float
    adj_p_high: float
    adj_predicted_class: str  # "low" | "medium" | "high"


def _sanitize_ticker(ticker: str) -> str:
    return _SAFE_TICKER_RE.sub("", ticker)[:30]


def _excess_kurtosis(arr: np.ndarray) -> float:
    """Fisher's excess kurtosis (normal distribution = 0, heavy tails > 0)."""
    n = len(arr)
    if n < 4:
        return 0.0
    mean = arr.mean()
    m2 = float(((arr - mean) ** 2).mean())
    m4 = float(((arr - mean) ** 4).mean())
    if m2 < 1e-12:
        return 0.0
    return m4 / m2 ** 2 - 3.0


def _tilt(p_low: float, p_medium: float, p_high: float, m: float) -> tuple[float, float, float]:
    """Shift regime probability mass toward high (m>1) or low (m<1)."""
    raw = [p_low / m, p_medium, p_high * m]
    total = sum(raw)
    if total < 1e-9:
        return p_low, p_medium, p_high
    adj = [r / total for r in raw]
    return adj[0], adj[1], adj[2]


def _compute_stats(
    closes_by_ticker: dict[str, list[float]],
) -> dict[str, tuple[float, float] | None]:
    """Return (annualized_vol, excess_kurtosis) per ticker, or None if insufficient data."""
    stats: dict[str, tuple[float, float] | None] = {}
    for ticker, closes in closes_by_ticker.items():
        arr = np.array(closes, dtype=float)
        rets = arr[1:] / arr[:-1] - 1.0
        if len(rets) < _MIN_OBS:
            stats[ticker] = None
            continue
        vol = float(np.std(rets, ddof=1) * np.sqrt(252))
        kurt = _excess_kurtosis(rets)
        stats[ticker] = (vol, kurt)
    return stats


def compute_risk_adjustments(
    research_db: duckdb.DuckDBPyConnection,
    positions: dict[str, tuple[str, float, float, float]],
    window_days: int = 504,
) -> dict[str, RiskAdjustment | None]:
    """Compute idiosyncratic risk adjustments for a set of portfolio positions.

    Parameters
    ----------
    research_db:
        Open DuckDB connection with access to the ``raw_prices`` table.
    positions:
        Mapping of ``{label: (proxy_source_ticker, p_low, p_medium, p_high)}``.
        ``label`` is the user-entered ticker symbol (e.g. ``"AAPL"``).
    window_days:
        Calendar-day lookback window (~2 years = 504).

    Returns
    -------
    dict[str, RiskAdjustment | None]
        ``None`` for positions where data is unavailable or insufficient.
    """
    if not positions:
        return {}

    # Collect all tickers to query — labels plus their proxy source tickers (deduped)
    all_tickers: set[str] = set()
    for label, (proxy_ticker, *_) in positions.items():
        safe_label = _sanitize_ticker(label)
        safe_proxy = _sanitize_ticker(proxy_ticker)
        if safe_label:
            all_tickers.add(safe_label)
        if safe_proxy:
            all_tickers.add(safe_proxy)

    if not all_tickers:
        return {label: None for label in positions}

    # Inline VALUES for the ticker list (sanitized above)
    values_sql = ", ".join(f"('{t}')" for t in all_tickers)

    try:
        rows = research_db.execute(f"""
            WITH ticker_list AS (
                SELECT column0 AS ticker
                FROM (VALUES {values_sql})
            )
            SELECT rp.ticker, rp.close
            FROM raw_prices rp
            JOIN ticker_list tl ON rp.ticker = tl.ticker
            WHERE rp.date >= CURRENT_DATE - INTERVAL ({window_days} || ' days')
              AND rp.close IS NOT NULL
            ORDER BY rp.ticker, rp.date
        """).fetchall()
    except Exception:
        logger.exception("risk_adjustment: DuckDB query failed; skipping adjustments")
        return {label: None for label in positions}

    # Group closes per ticker
    closes_by_ticker: dict[str, list[float]] = {}
    for ticker, close in rows:
        closes_by_ticker.setdefault(ticker, []).append(float(close))

    stats = _compute_stats(closes_by_ticker)

    results: dict[str, RiskAdjustment | None] = {}
    for label, (proxy_ticker, p_low, p_medium, p_high) in positions.items():
        safe_label = _sanitize_ticker(label)
        safe_proxy = _sanitize_ticker(proxy_ticker)

        asset_stats = stats.get(safe_label)
        proxy_stats = stats.get(safe_proxy)

        if asset_stats is None or proxy_stats is None:
            results[label] = None
            continue

        vol_asset, kurt_asset = asset_stats
        vol_proxy, kurt_proxy = proxy_stats

        if vol_proxy < 1e-6:
            results[label] = None
            continue

        vol_ratio = float(np.clip(vol_asset / vol_proxy, _CLAMP_MIN, _CLAMP_MAX))

        # Extra multiplier when the asset has fatter tails than the proxy
        kurt_excess_diff = kurt_asset - kurt_proxy
        kurt_denom = max(abs(kurt_proxy), 1.0)
        kurt_scale = 1.0 + _KURT_WEIGHT * max(0.0, kurt_excess_diff / kurt_denom)

        multiplier = float(np.clip(vol_ratio * kurt_scale, _CLAMP_MIN, _CLAMP_MAX))

        adj_low, adj_med, adj_high = _tilt(
            p_low or 0.0, p_medium or 0.0, p_high or 0.0, multiplier
        )
        adj_class = _CLASS_ORDER[
            [adj_low, adj_med, adj_high].index(max(adj_low, adj_med, adj_high))
        ]

        results[label] = RiskAdjustment(
            vol_asset=round(vol_asset, 6),
            vol_proxy=round(vol_proxy, 6),
            vol_ratio=round(vol_ratio, 4),
            kurt_scale=round(kurt_scale, 4),
            multiplier=round(multiplier, 4),
            adj_p_low=round(adj_low, 6),
            adj_p_medium=round(adj_med, 6),
            adj_p_high=round(adj_high, 6),
            adj_predicted_class=adj_class,
        )

    return results
