"""Chat tool definitions and executor functions.

Each tool is registered in TOOL_REGISTRY (name → callable).
TOOL_DEFINITIONS is the list of OpenAI tool schemas sent with every request.

Executor signature:
    (params: dict, user: User, db: Session, serving_db: ServingDB,
     research_db: DuckDBPyConnection) -> dict
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date, timedelta
from typing import Any

import duckdb
from sqlalchemy.orm import Session

from quant_risk.prod.assets import load_asset_catalog
from quant_risk.prod.auth.models import User
from quant_risk.prod.auth.portfolios import (
    PositionInput,
    create_portfolio,
    delete_portfolio,
    get_portfolio,
    list_user_portfolios,
    set_positions,
    update_portfolio_name,
)
from quant_risk.prod.portfolio_analysis import analyze_portfolio_orm
from quant_risk.prod.serving.duckdb import ServingDB
from quant_risk.prod.serving.predictions import (
    get_all_latest_predictions,
    get_latest_prediction,
    get_prediction_history,
)

logger = logging.getLogger(__name__)

_ASSETS_CONFIG = "config/prod/assets.yaml"

VALID_ASSET_IDS = {
    "us_equities",
    "euro_equities",
    "bitcoin",
    "long_us_treasuries",
    "short_us_treasuries",
    "gold",
}


# ---------------------------------------------------------------------------
# Executor functions
# ---------------------------------------------------------------------------

def _exec_get_latest_predictions(
    params: dict, user: User, db: Session,
    serving_db: ServingDB, research_db: duckdb.DuckDBPyConnection,
) -> dict:
    asset_id = params.get("asset_id")
    if asset_id:
        row = get_latest_prediction(serving_db, asset_id)
        if row is None:
            return {"error": f"No predictions found for '{asset_id}'."}
        return {"predictions": [_clean_prediction(row)]}
    rows = get_all_latest_predictions(serving_db)
    return {"predictions": [_clean_prediction(r) for r in rows]}


def _exec_get_prediction_history(
    params: dict, user: User, db: Session,
    serving_db: ServingDB, research_db: duckdb.DuckDBPyConnection,
) -> dict:
    asset_id = params["asset_id"]
    days = params.get("days", 30)
    rows = get_prediction_history(serving_db, asset_id, limit=days)
    if not rows:
        return {"error": f"No prediction history for '{asset_id}'."}
    return {"predictions": [_clean_prediction(r) for r in rows]}


def _exec_get_price_history(
    params: dict, user: User, db: Session,
    serving_db: ServingDB, research_db: duckdb.DuckDBPyConnection,
) -> dict:
    asset_id = params["asset_id"]
    days = params.get("days", 90)
    catalog = load_asset_catalog(_ASSETS_CONFIG)
    asset = next((a for a in catalog if a.asset_id == asset_id), None)
    if asset is None:
        return {"error": f"Asset '{asset_id}' not found."}
    since = date.today() - timedelta(days=days)
    rows = research_db.execute(
        """
        SELECT CAST(date AS VARCHAR) AS date, close
        FROM raw_prices
        WHERE ticker = ? AND date >= ? AND close IS NOT NULL
        ORDER BY date ASC
        """,
        [asset.source_ticker, since],
    ).fetchall()
    return {"prices": [{"date": r[0], "close": round(r[1], 4)} for r in rows]}


def _exec_get_regime_history(
    params: dict, user: User, db: Session,
    serving_db: ServingDB, research_db: duckdb.DuckDBPyConnection,
) -> dict:
    asset_id = params["asset_id"]
    days = params.get("days", 90)
    catalog = load_asset_catalog(_ASSETS_CONFIG)
    asset = next((a for a in catalog if a.asset_id == asset_id), None)
    if asset is None:
        return {"error": f"Asset '{asset_id}' not found."}
    since = date.today() - timedelta(days=days)
    rows = research_db.execute(
        """
        WITH thresholds AS (
            SELECT
                approx_quantile(vol_fwd, 0.333) AS q33,
                approx_quantile(vol_fwd, 0.667) AS q67
            FROM labels_regime
            WHERE ticker = ? AND horizon = 5
        )
        SELECT
            CAST(lr.date AS VARCHAR) AS date,
            CASE
                WHEN lr.vol_fwd <= t.q33 THEN 'low'
                WHEN lr.vol_fwd <= t.q67 THEN 'medium'
                ELSE 'high'
            END AS regime
        FROM labels_regime lr
        CROSS JOIN thresholds t
        WHERE lr.ticker = ? AND lr.horizon = 5 AND lr.date >= ?
        ORDER BY lr.date ASC
        """,
        [asset.source_ticker, asset.source_ticker, since],
    ).fetchall()
    return {"regimes": [{"date": r[0], "regime": r[1]} for r in rows]}


def _exec_get_vol_profile(
    params: dict, user: User, db: Session,
    serving_db: ServingDB, research_db: duckdb.DuckDBPyConnection,
) -> dict:
    asset_id = params["asset_id"]
    catalog = load_asset_catalog(_ASSETS_CONFIG)
    asset = next((a for a in catalog if a.asset_id == asset_id), None)
    if asset is None:
        return {"error": f"Asset '{asset_id}' not found."}
    row = research_db.execute(
        """
        WITH thresholds AS (
            SELECT
                approx_quantile(vol_fwd, 0.333) AS q33,
                approx_quantile(vol_fwd, 0.667) AS q67
            FROM labels_regime WHERE ticker = ? AND horizon = 5
        )
        SELECT
            (SELECT median(vol_fwd) FROM labels_regime WHERE ticker = ?
             AND horizon = 5 AND vol_fwd <= (SELECT q33 FROM thresholds)),
            (SELECT median(vol_fwd) FROM labels_regime lr, thresholds t
             WHERE lr.ticker = ? AND lr.horizon = 5
             AND lr.vol_fwd > t.q33 AND lr.vol_fwd <= t.q67),
            (SELECT median(vol_fwd) FROM labels_regime lr, thresholds t
             WHERE lr.ticker = ? AND lr.horizon = 5 AND lr.vol_fwd > t.q67)
        """,
        [asset.source_ticker, asset.source_ticker,
         asset.source_ticker, asset.source_ticker],
    ).fetchone()
    if row is None:
        return {"error": f"No vol data for '{asset_id}'."}
    return {
        "asset_id": asset_id,
        "vol_5d_low": round(row[0], 6) if row[0] else None,
        "vol_5d_medium": round(row[1], 6) if row[1] else None,
        "vol_5d_high": round(row[2], 6) if row[2] else None,
    }


def _exec_list_portfolios(
    params: dict, user: User, db: Session,
    serving_db: ServingDB, research_db: duckdb.DuckDBPyConnection,
) -> dict:
    portfolios = list_user_portfolios(db, user.id)
    return {
        "portfolios": [
            {
                "portfolio_id": p.id,
                "name": p.name,
                "position_count": len(p.positions),
                "total_weight_pct": sum(pos.weight_pct for pos in p.positions),
                "last_signal": p.last_analysis_signal,
            }
            for p in portfolios
        ]
    }


def _exec_get_portfolio(
    params: dict, user: User, db: Session,
    serving_db: ServingDB, research_db: duckdb.DuckDBPyConnection,
) -> dict:
    portfolio_id = params["portfolio_id"]
    try:
        p = get_portfolio(db, portfolio_id, user.id)
    except Exception:
        return {"error": f"Portfolio '{portfolio_id}' not found."}
    return {
        "portfolio_id": p.id,
        "name": p.name,
        "positions": [
            {"label": pos.label, "weight_pct": pos.weight_pct,
             "proxy_asset_id": pos.proxy_asset_id}
            for pos in p.positions
        ],
        "last_signal": p.last_analysis_signal,
    }


def _exec_analyze_portfolio(
    params: dict, user: User, db: Session,
    serving_db: ServingDB, research_db: duckdb.DuckDBPyConnection,
) -> dict:
    portfolio_id = params["portfolio_id"]
    try:
        portfolio = get_portfolio(db, portfolio_id, user.id)
    except Exception:
        return {"error": f"Portfolio '{portfolio_id}' not found."}
    try:
        result = analyze_portfolio_orm(portfolio, serving_db)
    except ValueError as exc:
        return {"error": str(exc)}
    return _analysis_to_dict(result)


def _exec_create_portfolio(
    params: dict, user: User, db: Session,
    serving_db: ServingDB, research_db: duckdb.DuckDBPyConnection,
) -> dict:
    name = params["name"]
    positions = params.get("positions", [])
    for pos in positions:
        if pos.get("proxy_asset_id") not in VALID_ASSET_IDS:
            return {"error": f"Invalid proxy_asset_id: '{pos.get('proxy_asset_id')}'. "
                    f"Valid IDs: {sorted(VALID_ASSET_IDS)}"}
    try:
        portfolio = create_portfolio(
            db, user_id=user.id, name=name,
            positions=[
                PositionInput(
                    label=p["label"],
                    weight_pct=p["weight_pct"],
                    proxy_asset_id=p["proxy_asset_id"],
                )
                for p in positions
            ],
        )
        db.commit()
        db.refresh(portfolio)
    except Exception as exc:
        db.rollback()
        return {"error": str(exc)}
    return {"portfolio_id": portfolio.id, "name": portfolio.name,
            "position_count": len(portfolio.positions)}


def _exec_update_portfolio(
    params: dict, user: User, db: Session,
    serving_db: ServingDB, research_db: duckdb.DuckDBPyConnection,
) -> dict:
    portfolio_id = params["portfolio_id"]
    try:
        get_portfolio(db, portfolio_id, user.id)
    except Exception:
        return {"error": f"Portfolio '{portfolio_id}' not found."}
    try:
        if "name" in params and params["name"] is not None:
            update_portfolio_name(db, portfolio_id, user.id, params["name"])
        if "positions" in params and params["positions"] is not None:
            for pos in params["positions"]:
                if pos.get("proxy_asset_id") not in VALID_ASSET_IDS:
                    return {"error": f"Invalid proxy_asset_id: '{pos.get('proxy_asset_id')}'."}
            set_positions(
                db, portfolio_id, user.id,
                [PositionInput(label=p["label"], weight_pct=p["weight_pct"],
                               proxy_asset_id=p["proxy_asset_id"])
                 for p in params["positions"]],
            )
        db.commit()
        portfolio = get_portfolio(db, portfolio_id, user.id)
    except Exception as exc:
        db.rollback()
        return {"error": str(exc)}
    return {"portfolio_id": portfolio.id, "name": portfolio.name,
            "position_count": len(portfolio.positions), "status": "updated"}


def _exec_whatif_analysis(
    params: dict, user: User, db: Session,
    serving_db: ServingDB, research_db: duckdb.DuckDBPyConnection,
) -> dict:
    portfolio_id = params["portfolio_id"]
    positions = params.get("positions", [])
    for pos in positions:
        if pos.get("proxy_asset_id") not in VALID_ASSET_IDS:
            return {"error": f"Invalid proxy_asset_id: '{pos.get('proxy_asset_id')}'."}
    try:
        portfolio = get_portfolio(db, portfolio_id, user.id)
    except Exception:
        return {"error": f"Portfolio '{portfolio_id}' not found."}
    overrides = [
        PositionInput(label=p["label"], weight_pct=p["weight_pct"],
                      proxy_asset_id=p["proxy_asset_id"])
        for p in positions
    ]
    try:
        result = analyze_portfolio_orm(portfolio, serving_db, position_overrides=overrides)
    except ValueError as exc:
        return {"error": str(exc)}
    return _analysis_to_dict(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_prediction(row: dict) -> dict:
    """Strip internal fields from a prediction row for LLM consumption."""
    return {
        "asset_id": row["asset_id"],
        "forecast_date": str(row["forecast_date"]),
        "predicted_class": row["predicted_class"],
        "p_low": round(row["p_low"], 4),
        "p_medium": round(row["p_medium"], 4),
        "p_high": round(row["p_high"], 4),
    }


def _analysis_to_dict(result) -> dict:
    """Convert a PortfolioAnalysis dataclass to a JSON-friendly dict."""
    return {
        "portfolio_signal": result.portfolio_signal,
        "portfolio_p_low": round(result.portfolio_p_low, 4),
        "portfolio_p_medium": round(result.portfolio_p_medium, 4),
        "portfolio_p_high": round(result.portfolio_p_high, 4),
        "total_weight_pct": result.total_weight_pct,
        "missing_predictions": result.missing_predictions,
        "asset_groups": [
            {
                "asset_id": ag.asset_id,
                "aggregate_weight": round(ag.aggregate_weight, 4),
                "predicted_class": ag.predicted_class,
                "p_low": round(ag.p_low, 4) if ag.p_low is not None else None,
                "p_medium": round(ag.p_medium, 4) if ag.p_medium is not None else None,
                "p_high": round(ag.p_high, 4) if ag.p_high is not None else None,
            }
            for ag in result.asset_groups
        ],
    }


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_latest_predictions",
            "description": "Get the latest volatility regime predictions. Omit asset_id to get all assets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Asset ID (e.g. 'us_equities', 'bitcoin'). Omit for all assets.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_prediction_history",
            "description": "Get historical volatility predictions for one asset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Asset ID (e.g. 'us_equities', 'bitcoin').",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days of history (default 30).",
                    },
                },
                "required": ["asset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": "Get daily close prices for one asset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Asset ID (e.g. 'us_equities', 'bitcoin').",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days of history (default 90).",
                    },
                },
                "required": ["asset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_regime_history",
            "description": "Get realised volatility regime history for one asset (based on actual vol, not predictions).",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Asset ID (e.g. 'us_equities', 'bitcoin').",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days of history (default 90).",
                    },
                },
                "required": ["asset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vol_profile",
            "description": "Get the median 5-day realised volatility for each regime tier (low/medium/high) for one asset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Asset ID (e.g. 'us_equities', 'bitcoin').",
                    },
                },
                "required": ["asset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_portfolios",
            "description": "List all portfolios belonging to the current user.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": "Get full detail of a portfolio including all positions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {
                        "type": "string",
                        "description": "Portfolio UUID.",
                    },
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_portfolio",
            "description": "Run full analysis on a saved portfolio against current volatility predictions. Returns signal, probabilities, and per-asset breakdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {
                        "type": "string",
                        "description": "Portfolio UUID.",
                    },
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_portfolio",
            "description": "Create a new portfolio with positions. Each position needs a label, weight_pct (percentage), and proxy_asset_id (one of: us_equities, euro_equities, bitcoin, gold, long_us_treasuries, short_us_treasuries).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Portfolio name.",
                    },
                    "positions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "weight_pct": {"type": "number"},
                                "proxy_asset_id": {"type": "string"},
                            },
                            "required": ["label", "weight_pct", "proxy_asset_id"],
                        },
                        "description": "List of positions.",
                    },
                },
                "required": ["name", "positions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_portfolio",
            "description": "Update an existing portfolio's name and/or positions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {
                        "type": "string",
                        "description": "Portfolio UUID.",
                    },
                    "name": {
                        "type": "string",
                        "description": "New name (optional).",
                    },
                    "positions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "weight_pct": {"type": "number"},
                                "proxy_asset_id": {"type": "string"},
                            },
                            "required": ["label", "weight_pct", "proxy_asset_id"],
                        },
                        "description": "New positions (replaces all existing).",
                    },
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "whatif_analysis",
            "description": "Run a what-if analysis with hypothetical positions without modifying the saved portfolio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {
                        "type": "string",
                        "description": "Portfolio UUID (used for context/name).",
                    },
                    "positions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "weight_pct": {"type": "number"},
                                "proxy_asset_id": {"type": "string"},
                            },
                            "required": ["label", "weight_pct", "proxy_asset_id"],
                        },
                        "description": "Hypothetical positions to analyse.",
                    },
                },
                "required": ["portfolio_id", "positions"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    "get_latest_predictions": _exec_get_latest_predictions,
    "get_prediction_history": _exec_get_prediction_history,
    "get_price_history": _exec_get_price_history,
    "get_regime_history": _exec_get_regime_history,
    "get_vol_profile": _exec_get_vol_profile,
    "list_portfolios": _exec_list_portfolios,
    "get_portfolio": _exec_get_portfolio,
    "analyze_portfolio": _exec_analyze_portfolio,
    "create_portfolio": _exec_create_portfolio,
    "update_portfolio": _exec_update_portfolio,
    "whatif_analysis": _exec_whatif_analysis,
}
