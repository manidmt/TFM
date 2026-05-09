'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Private portfolio endpoints — requires authentication (rpi5.md §12.2, §15).

Routes
------
GET  /api/private/portfolios
    List the current user's portfolios (name + id, no positions).

POST /api/private/portfolios
    Create a new portfolio with positions.

GET  /api/private/portfolios/{portfolio_id}
    Get one portfolio with all positions.

PUT  /api/private/portfolios/{portfolio_id}
    Update portfolio name and/or replace all positions atomically.

DELETE /api/private/portfolios/{portfolio_id}
    Delete a portfolio and all its positions.

POST /api/private/portfolios/{portfolio_id}/analyze
    Run the portfolio analysis pipeline and return the result.
    Accepts an optional ``positions`` override for what-if analysis.
'''

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from quant_risk.prod.api.deps import get_auth_db, get_current_user, get_serving_db
from quant_risk.prod.auth.models import User
from quant_risk.prod.auth.portfolios import (
    PortfolioNotFoundError,
    PortfolioValidationError,
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
from quant_risk.prod.var_engine import VaRResult, compute_portfolio_var
from quant_risk.prod.assets import load_asset_catalog

import duckdb
from fastapi.responses import StreamingResponse

from quant_risk.prod.api.deps import get_config, get_research_db
from quant_risk.prod.api.config import AppConfig
from quant_risk.prod.chat.service import chat_stream

router = APIRouter(prefix="/api/private", tags=["private"])

_ASSETS_CONFIG = "config/prod/assets.yaml"


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PositionIn(BaseModel):
    label: str
    weight_pct: float
    proxy_asset_id: str


class CreatePortfolioRequest(BaseModel):
    name: str
    positions: list[PositionIn]


class UpdatePortfolioRequest(BaseModel):
    name: str | None = None
    positions: list[PositionIn] | None = None


class PortfolioSummaryOut(BaseModel):
    portfolio_id: str
    name: str
    position_count: int
    total_weight_pct: float
    last_signal: str | None
    last_analysis_at: str | None


class PositionOut(BaseModel):
    label: str
    weight_pct: float
    proxy_asset_id: str


class PortfolioDetailOut(BaseModel):
    portfolio_id: str
    name: str
    positions: list[PositionOut]
    last_signal: str | None
    last_analysis_at: str | None


class PositionAnalysisOut(BaseModel):
    label: str
    weight_pct: float
    normalized_weight: float
    proxy_asset_id: str
    predicted_class: str | None
    p_low: float | None
    p_medium: float | None
    p_high: float | None
    forecast_date: str | None


class AssetGroupOut(BaseModel):
    asset_id: str
    aggregate_weight: float
    predicted_class: str | None
    p_low: float | None
    p_medium: float | None
    p_high: float | None
    forecast_date: str | None
    position_labels: list[str]


class PortfolioAnalysisOut(BaseModel):
    portfolio_id: str
    portfolio_name: str
    total_weight_pct: float
    positions: list[PositionAnalysisOut]
    asset_groups: list[AssetGroupOut]
    portfolio_p_low: float
    portfolio_p_medium: float
    portfolio_p_high: float
    portfolio_signal: str | None
    missing_predictions: list[str]


class AnalyzeRequest(BaseModel):
    positions: list[PositionIn] | None = None


class VaRRequest(BaseModel):
    regime: str | None = None  # "low" | "medium" | "high" | "all"; None → use last signal


class PortfolioVaROut(BaseModel):
    var_95: float
    var_99: float
    cvar_95: float
    regime_filter: str
    scenario_count: int
    low_sample_warning: bool
    histogram: list[float]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., max_length=50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos_in_to_input(p: PositionIn) -> PositionInput:
    return PositionInput(
        label=p.label,
        weight_pct=p.weight_pct,
        proxy_asset_id=p.proxy_asset_id,
    )


def _portfolio_detail_out(portfolio) -> PortfolioDetailOut:
    return PortfolioDetailOut(
        portfolio_id=portfolio.id,
        name=portfolio.name,
        positions=[
            PositionOut(
                label=pos.label,
                weight_pct=pos.weight_pct,
                proxy_asset_id=pos.proxy_asset_id,
            )
            for pos in portfolio.positions
        ],
        last_signal=portfolio.last_analysis_signal,
        last_analysis_at=str(portfolio.last_analysis_at) if portfolio.last_analysis_at else None,
    )


def _handle_portfolio_errors(exc: Exception) -> None:
    if isinstance(exc, PortfolioNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, PortfolioValidationError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    raise exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/portfolios", response_model=list[PortfolioSummaryOut])
def list_portfolios(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    """List the current user's portfolios."""
    portfolios = list_user_portfolios(db, current_user.id)
    return [
        PortfolioSummaryOut(
            portfolio_id=p.id,
            name=p.name,
            position_count=len(p.positions),
            total_weight_pct=sum(pos.weight_pct for pos in p.positions),
            last_signal=p.last_analysis_signal,
            last_analysis_at=str(p.last_analysis_at) if p.last_analysis_at else None,
        )
        for p in portfolios
    ]


@router.post(
    "/portfolios",
    response_model=PortfolioDetailOut,
    status_code=status.HTTP_201_CREATED,
)
def create_portfolio_endpoint(
    body: CreatePortfolioRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    """Create a new portfolio with positions."""
    try:
        portfolio = create_portfolio(
            db,
            user_id=current_user.id,
            name=body.name,
            positions=[_pos_in_to_input(p) for p in body.positions],
        )
        db.commit()
        db.refresh(portfolio)
    except (PortfolioNotFoundError, PortfolioValidationError) as exc:
        _handle_portfolio_errors(exc)

    return _portfolio_detail_out(portfolio)


@router.get("/portfolios/{portfolio_id}", response_model=PortfolioDetailOut)
def get_portfolio_endpoint(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    """Get one portfolio with all positions."""
    try:
        portfolio = get_portfolio(db, portfolio_id, current_user.id)
    except PortfolioNotFoundError as exc:
        _handle_portfolio_errors(exc)

    return _portfolio_detail_out(portfolio)


@router.put("/portfolios/{portfolio_id}", response_model=PortfolioDetailOut)
def update_portfolio_endpoint(
    portfolio_id: str,
    body: UpdatePortfolioRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    """Update portfolio name and/or replace all positions atomically."""
    if body.name is None and body.positions is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of 'name' or 'positions' must be provided.",
        )
    try:
        if body.name is not None:
            update_portfolio_name(db, portfolio_id, current_user.id, body.name)
        if body.positions is not None:
            set_positions(
                db,
                portfolio_id,
                current_user.id,
                [_pos_in_to_input(p) for p in body.positions],
            )
        db.commit()
        portfolio = get_portfolio(db, portfolio_id, current_user.id)
    except (PortfolioNotFoundError, PortfolioValidationError) as exc:
        _handle_portfolio_errors(exc)

    return _portfolio_detail_out(portfolio)


@router.delete("/portfolios/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_portfolio_endpoint(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    """Delete a portfolio and all its positions."""
    try:
        delete_portfolio(db, portfolio_id, current_user.id)
        db.commit()
    except PortfolioNotFoundError as exc:
        _handle_portfolio_errors(exc)


@router.post("/portfolios/{portfolio_id}/analyze", response_model=PortfolioAnalysisOut)
def analyze_portfolio_endpoint(
    portfolio_id: str,
    body: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
    serving_db: ServingDB = Depends(get_serving_db),
):
    """Run portfolio analysis.

    Pass ``positions`` in the request body for what-if analysis without
    saving.  Omit ``positions`` to analyse the saved portfolio state.
    """
    try:
        portfolio = get_portfolio(db, portfolio_id, current_user.id)
    except PortfolioNotFoundError as exc:
        _handle_portfolio_errors(exc)

    overrides = (
        [_pos_in_to_input(p) for p in body.positions]
        if body.positions is not None
        else None
    )

    try:
        result = analyze_portfolio_orm(portfolio, serving_db, position_overrides=overrides)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )

    # Persist last analysis metadata (only for real analysis, not what-if)
    if body.positions is None:
        portfolio.last_analysis_signal = result.portfolio_signal
        portfolio.last_analysis_at = datetime.now(timezone.utc)
        db.commit()

    return PortfolioAnalysisOut(
        portfolio_id=result.portfolio_id,
        portfolio_name=result.portfolio_name,
        total_weight_pct=result.total_weight_pct,
        positions=[
            PositionAnalysisOut(
                label=pa.label,
                weight_pct=pa.weight_pct,
                normalized_weight=pa.normalized_weight,
                proxy_asset_id=pa.proxy_asset_id,
                predicted_class=pa.predicted_class,
                p_low=pa.p_low,
                p_medium=pa.p_medium,
                p_high=pa.p_high,
                forecast_date=str(pa.forecast_date) if pa.forecast_date else None,
            )
            for pa in result.positions
        ],
        asset_groups=[
            AssetGroupOut(
                asset_id=ag.asset_id,
                aggregate_weight=ag.aggregate_weight,
                predicted_class=ag.predicted_class,
                p_low=ag.p_low,
                p_medium=ag.p_medium,
                p_high=ag.p_high,
                forecast_date=str(ag.forecast_date) if ag.forecast_date else None,
                position_labels=ag.position_labels,
            )
            for ag in result.asset_groups
        ],
        portfolio_p_low=result.portfolio_p_low,
        portfolio_p_medium=result.portfolio_p_medium,
        portfolio_p_high=result.portfolio_p_high,
        portfolio_signal=result.portfolio_signal,
        missing_predictions=result.missing_predictions,
    )


@router.post("/portfolios/{portfolio_id}/var", response_model=PortfolioVaROut)
def portfolio_var(
    portfolio_id: str,
    body: VaRRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
    research_db: duckdb.DuckDBPyConnection = Depends(get_research_db),
):
    """Compute regime-conditioned 1-day Historical Simulation VaR for a portfolio."""
    try:
        portfolio = get_portfolio(db, portfolio_id, current_user.id)
    except PortfolioNotFoundError as exc:
        _handle_portfolio_errors(exc)

    if not portfolio.positions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Portfolio has no positions.",
        )

    # Determine regime filter: explicit body param → last analysis signal → "all"
    regime_filter = body.regime or portfolio.last_analysis_signal or "all"
    if regime_filter not in ("low", "medium", "high", "all"):
        regime_filter = "all"

    # Aggregate positions by proxy_asset_id and normalize weights
    asset_weights: dict[str, float] = {}
    for pos in portfolio.positions:
        asset_weights[pos.proxy_asset_id] = (
            asset_weights.get(pos.proxy_asset_id, 0.0) + pos.weight_pct
        )
    total_w = sum(asset_weights.values())
    if total_w > 0:
        asset_weights = {k: v / total_w for k, v in asset_weights.items()}

    # Map proxy_asset_id → source_ticker
    try:
        catalog = load_asset_catalog(_ASSETS_CONFIG)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Asset catalog not available.",
        )
    ticker_map = {a.asset_id: a.source_ticker for a in catalog}
    ticker_weights = [
        (ticker_map[aid], w)
        for aid, w in asset_weights.items()
        if aid in ticker_map
    ]

    if not ticker_weights:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="None of the portfolio positions map to a known asset.",
        )

    try:
        result: VaRResult = compute_portfolio_var(research_db, ticker_weights, regime_filter)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )

    return PortfolioVaROut(
        var_95=result.var_95,
        var_99=result.var_99,
        cvar_95=result.cvar_95,
        regime_filter=result.regime_filter,
        scenario_count=result.scenario_count,
        low_sample_warning=result.low_sample_warning,
        histogram=result.histogram,
    )


@router.post("/chat")
def chat_endpoint(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    config: AppConfig = Depends(get_config),
    db: Session = Depends(get_auth_db),
    serving_db: ServingDB = Depends(get_serving_db),
    research_db: duckdb.DuckDBPyConnection = Depends(get_research_db),
):
    """Stream a chat response using OpenAI with tool use."""
    if not config.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat is not configured.",
        )

    # Validate message content length
    for msg in body.messages:
        if len(msg.content) > 2000:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Message content must be under 2000 characters.",
            )

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    return StreamingResponse(
        chat_stream(
            messages=messages,
            user=current_user,
            db=db,
            serving_db=serving_db,
            research_db=research_db,
            api_key=config.openai_api_key,
            model=config.openai_model,
        ),
        media_type="text/event-stream",
    )
