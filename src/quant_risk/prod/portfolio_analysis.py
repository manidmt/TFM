'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Portfolio analysis service (rpi5.md §15.5).

This module is the **backend computation layer** for portfolio analysis.
It is pure business logic — no HTTP, no DB session management.

Analysis pipeline (rpi5.md §15.5):
  1. Validate and normalise weights (long-only, sum to 1.0)
  2. Aggregate normalised weights by proxy ``asset_id``
  3. Join the latest production prediction for each proxy asset
  4. Compute weighted average probability distribution across all assets
  5. Derive the portfolio-level signal (dominant class by weighted probability)

What-if analysis:
  Call ``analyze_portfolio`` with a custom ``positions`` override to simulate
  a modified allocation without saving it to the database.

Output contract:
  - ``PortfolioAnalysis`` is the top-level result dataclass
  - Predictions are sourced from ``serving.duckdb`` via ``ServingDB``
  - Assets with no available prediction are excluded from the weighted signal
    and reported in ``missing_predictions``

Design constraints (rpi5.md §15.6):
  - V1 is NOT a Monte Carlo engine or institutional risk system
  - The main output is predicted class + weighted probability distribution
  - Weights are expressed as fractions of the total portfolio (not absolute $)
'''

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from quant_risk.prod.auth.models import Portfolio, PortfolioPosition
from quant_risk.prod.auth.portfolios import PositionInput
from quant_risk.prod.serving.duckdb import ServingDB

logger = logging.getLogger(__name__)

# Canonical class ordering — matches PredictedClass enum in inference.py
_CLASS_ORDER = ("low", "medium", "high")


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PositionAnalysis:
    """Analysis result for one portfolio position.

    Attributes
    ----------
    label:
        Freeform label supplied by the user (e.g. ``AAPL``).
    weight_pct:
        Raw weight as supplied by the user (percent).
    normalized_weight:
        Weight normalised so all positions sum to 1.0.
    proxy_asset_id:
        Production asset ID used as the volatility proxy.
    predicted_class:
        Latest predicted volatility class, or None if unavailable.
    p_low / p_medium / p_high:
        Class probabilities from the latest prediction, or None.
    forecast_date:
        Date of the prediction used, or None.
    """

    label: str
    weight_pct: float
    normalized_weight: float
    proxy_asset_id: str
    predicted_class: str | None
    p_low: float | None
    p_medium: float | None
    p_high: float | None
    forecast_date: date | None


@dataclass
class AssetGroupAnalysis:
    """Aggregated analysis for all positions mapped to one proxy asset.

    Attributes
    ----------
    asset_id:
        Production asset ID.
    aggregate_weight:
        Sum of normalised weights for all positions mapped to this asset.
    predicted_class:
        Latest predicted class for this asset, or None.
    p_low / p_medium / p_high:
        Class probabilities, or None.
    forecast_date:
        Date of the prediction, or None.
    position_labels:
        Labels of the constituent positions.
    """

    asset_id: str
    aggregate_weight: float
    predicted_class: str | None
    p_low: float | None
    p_medium: float | None
    p_high: float | None
    forecast_date: date | None
    position_labels: list[str] = field(default_factory=list)


@dataclass
class PortfolioAnalysis:
    """Top-level portfolio analysis result.

    Attributes
    ----------
    portfolio_id:
        ID of the portfolio that was analysed (empty string for ad-hoc what-if).
    portfolio_name:
        Name of the portfolio.
    total_weight_pct:
        Sum of raw ``weight_pct`` values before normalisation.
    positions:
        Per-position analysis, including prediction join.
    asset_groups:
        Aggregated analysis grouped by ``proxy_asset_id``.
    portfolio_p_low / portfolio_p_medium / portfolio_p_high:
        Weighted average class probabilities across all assets with predictions.
        Each is in [0, 1]; they sum to 1.0 when all assets have predictions.
        If no predictions are available, all three are 0.0.
    portfolio_signal:
        Dominant volatility class by weighted probability
        (``'low'``, ``'medium'``, or ``'high'``).
        ``None`` when no predictions are available for any asset.
    missing_predictions:
        ``asset_id`` values for which no prediction was found in
        ``serving.duckdb``.  Assets in this list are excluded from the
        weighted signal computation.
    """

    portfolio_id: str
    portfolio_name: str
    total_weight_pct: float
    positions: list[PositionAnalysis]
    asset_groups: list[AssetGroupAnalysis]
    portfolio_p_low: float
    portfolio_p_medium: float
    portfolio_p_high: float
    portfolio_signal: str | None
    missing_predictions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_weights(positions: list[PositionInput]) -> list[float]:
    """Return normalised weights that sum to 1.0.

    If the total weight is zero (all weights are 0), each position receives
    an equal share (1 / n).
    """
    total = sum(p.weight_pct for p in positions)
    if total == 0:
        n = len(positions)
        return [1.0 / n] * n
    return [p.weight_pct / total for p in positions]


def _fetch_predictions(
    serving_db: ServingDB,
    asset_ids: set[str],
) -> dict[str, dict]:
    """Return a mapping of asset_id → latest prediction row for each asset in *asset_ids*.

    Missing assets are absent from the result dict.
    """
    from quant_risk.prod.serving.predictions import get_all_latest_predictions

    all_preds = get_all_latest_predictions(serving_db)
    return {row["asset_id"]: row for row in all_preds if row["asset_id"] in asset_ids}


def _parse_date(value) -> date | None:
    """Parse a forecast_date value that may be a string, date, or None."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_portfolio(
    portfolio_id: str,
    portfolio_name: str,
    positions: list[PositionInput],
    serving_db: ServingDB,
) -> PortfolioAnalysis:
    """Run the full portfolio analysis pipeline.

    Parameters
    ----------
    portfolio_id:
        ID of the portfolio (pass empty string for ad-hoc what-if analysis).
    portfolio_name:
        Display name.
    positions:
        List of ``PositionInput`` (label, weight_pct, proxy_asset_id).
        Must not be empty.
    serving_db:
        Open ``ServingDB`` connection for reading latest predictions.

    Returns
    -------
    PortfolioAnalysis
        Always returns — never raises.  Check ``missing_predictions`` for
        assets without available predictions.

    Raises
    ------
    ValueError
        If *positions* is empty.
    """
    if not positions:
        raise ValueError("Cannot analyse a portfolio with no positions.")

    total_weight_pct = sum(p.weight_pct for p in positions)
    normalised = _normalize_weights(positions)

    # 1. Fetch predictions for all proxy asset IDs
    proxy_ids = {p.proxy_asset_id for p in positions}
    pred_map = _fetch_predictions(serving_db, proxy_ids)
    missing = sorted(proxy_ids - pred_map.keys())

    # 2. Build per-position analysis
    position_analyses: list[PositionAnalysis] = []
    for pos, norm_w in zip(positions, normalised):
        pred = pred_map.get(pos.proxy_asset_id)
        position_analyses.append(
            PositionAnalysis(
                label=pos.label,
                weight_pct=pos.weight_pct,
                normalized_weight=norm_w,
                proxy_asset_id=pos.proxy_asset_id,
                predicted_class=pred.get("predicted_class") if pred else None,
                p_low=pred.get("p_low") if pred else None,
                p_medium=pred.get("p_medium") if pred else None,
                p_high=pred.get("p_high") if pred else None,
                forecast_date=_parse_date(pred.get("forecast_date")) if pred else None,
            )
        )

    # 3. Aggregate by asset_id
    asset_agg: dict[str, dict] = {}
    for pa in position_analyses:
        aid = pa.proxy_asset_id
        if aid not in asset_agg:
            pred = pred_map.get(aid)
            asset_agg[aid] = {
                "aggregate_weight": 0.0,
                "predicted_class": pred.get("predicted_class") if pred else None,
                "p_low": pred.get("p_low") if pred else None,
                "p_medium": pred.get("p_medium") if pred else None,
                "p_high": pred.get("p_high") if pred else None,
                "forecast_date": _parse_date(pred.get("forecast_date")) if pred else None,
                "position_labels": [],
            }
        asset_agg[aid]["aggregate_weight"] += pa.normalized_weight
        asset_agg[aid]["position_labels"].append(pa.label)

    asset_groups = [
        AssetGroupAnalysis(
            asset_id=aid,
            aggregate_weight=v["aggregate_weight"],
            predicted_class=v["predicted_class"],
            p_low=v["p_low"],
            p_medium=v["p_medium"],
            p_high=v["p_high"],
            forecast_date=v["forecast_date"],
            position_labels=v["position_labels"],
        )
        for aid, v in sorted(asset_agg.items())
    ]

    # 4. Compute portfolio-level weighted signal (exclude assets without predictions)
    covered_groups = [g for g in asset_groups if g.predicted_class is not None]

    if not covered_groups:
        port_p_low = port_p_medium = port_p_high = 0.0
        port_signal = None
    else:
        # Re-normalise weights among covered assets only
        covered_weight_total = sum(g.aggregate_weight for g in covered_groups)
        port_p_low = port_p_medium = port_p_high = 0.0
        for g in covered_groups:
            w = g.aggregate_weight / covered_weight_total
            port_p_low += w * (g.p_low or 0.0)
            port_p_medium += w * (g.p_medium or 0.0)
            port_p_high += w * (g.p_high or 0.0)

        # Dominant class
        probs = (port_p_low, port_p_medium, port_p_high)
        port_signal = _CLASS_ORDER[probs.index(max(probs))]

    logger.info(
        "Portfolio analysis complete: id=%s signal=%s missing=%s",
        portfolio_id, port_signal, missing,
    )
    return PortfolioAnalysis(
        portfolio_id=portfolio_id,
        portfolio_name=portfolio_name,
        total_weight_pct=total_weight_pct,
        positions=position_analyses,
        asset_groups=asset_groups,
        portfolio_p_low=round(port_p_low, 6),
        portfolio_p_medium=round(port_p_medium, 6),
        portfolio_p_high=round(port_p_high, 6),
        portfolio_signal=port_signal,
        missing_predictions=missing,
    )


# ---------------------------------------------------------------------------
# Convenience: analyse a Portfolio ORM object
# ---------------------------------------------------------------------------

def analyze_portfolio_orm(
    portfolio: Portfolio,
    serving_db: ServingDB,
    *,
    position_overrides: list[PositionInput] | None = None,
) -> PortfolioAnalysis:
    """Analyse a loaded Portfolio ORM object.

    Parameters
    ----------
    portfolio:
        A loaded ``Portfolio`` instance (with ``positions`` relationship eager-
        or lazy-loaded).
    serving_db:
        Open ``ServingDB`` connection.
    position_overrides:
        If provided, use these positions instead of ``portfolio.positions``.
        Useful for what-if analysis without saving changes.

    Returns
    -------
    PortfolioAnalysis
    """
    if position_overrides is not None:
        positions = position_overrides
    else:
        positions = [
            PositionInput(
                label=p.label,
                weight_pct=p.weight_pct,
                proxy_asset_id=p.proxy_asset_id,
            )
            for p in portfolio.positions
        ]

    return analyze_portfolio(
        portfolio_id=portfolio.id,
        portfolio_name=portfolio.name,
        positions=positions,
        serving_db=serving_db,
    )
