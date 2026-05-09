'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Portfolio and position CRUD operations (rpi5.md §15.2, §15.4).

Portfolios are stored server-side in Postgres.  Each user may own multiple
portfolios.  V1 stores only the **current state** — no version history.

Position replacement is atomic: ``set_positions`` deletes all existing
positions for a portfolio and inserts the new set in one transaction.
This keeps the API simple and avoids partial-update edge cases.

Validation rules enforced here (rpi5.md §15.3):
  - portfolio name must be non-empty
  - each position label must be non-empty
  - weight_pct must be ≥ 0 (negative weights are not allowed)
  - proxy_asset_id must be non-empty
  - at least one position is required to set_positions

Weight normalisation (to ~100%) is done in the analysis service, not here,
so that CRUD does not impose a hard constraint on the raw input values.
'''

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from quant_risk.prod.auth.models import Portfolio, PortfolioPosition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PortfolioNotFoundError(LookupError):
    """Raised when a portfolio cannot be found or does not belong to the user."""


class PortfolioValidationError(ValueError):
    """Raised when portfolio or position data fails validation."""


# ---------------------------------------------------------------------------
# Input dataclass
# ---------------------------------------------------------------------------

@dataclass
class PositionInput:
    """One position supplied by the caller when creating/updating a portfolio.

    Attributes
    ----------
    label:
        Freeform ticker/label (e.g. ``AAPL``, ``BBVA``).
    weight_pct:
        Weight in percent (must be ≥ 0).
    proxy_asset_id:
        Production asset ID (e.g. ``us_equities``).  The analysis service
        validates that a prediction exists for this ID at query time.
    """

    label: str
    weight_pct: float
    proxy_asset_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_positions(positions: list[PositionInput], *, require_nonempty: bool = True) -> None:
    if require_nonempty and not positions:
        raise PortfolioValidationError("Portfolio must have at least one position.")
    for i, pos in enumerate(positions):
        if not pos.label.strip():
            raise PortfolioValidationError(
                f"Position {i}: label must not be empty."
            )
        if pos.weight_pct < 0:
            raise PortfolioValidationError(
                f"Position {i} ({pos.label!r}): weight_pct must be ≥ 0, "
                f"got {pos.weight_pct}."
            )
        if not pos.proxy_asset_id.strip():
            raise PortfolioValidationError(
                f"Position {i} ({pos.label!r}): proxy_asset_id must not be empty."
            )


def _validate_name(name: str) -> None:
    if not name.strip():
        raise PortfolioValidationError("Portfolio name must not be empty.")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_portfolio(db: Session, portfolio_id: str, user_id: str) -> Portfolio:
    """Return the portfolio, ensuring it belongs to *user_id*.

    Raises
    ------
    PortfolioNotFoundError
        If the portfolio does not exist or belongs to a different user.
    """
    p = db.execute(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == user_id,
        )
    ).scalar_one_or_none()
    if p is None:
        raise PortfolioNotFoundError(
            f"Portfolio '{portfolio_id}' not found for user '{user_id}'."
        )
    return p


def list_user_portfolios(db: Session, user_id: str) -> list[Portfolio]:
    """Return all portfolios owned by *user_id*, ordered by name."""
    return (
        db.execute(
            select(Portfolio)
            .where(Portfolio.user_id == user_id)
            .order_by(Portfolio.name)
        )
        .scalars()
        .all()
    )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def create_portfolio(
    db: Session,
    user_id: str,
    name: str,
    positions: list[PositionInput],
) -> Portfolio:
    """Create a new portfolio with its initial positions.

    Parameters
    ----------
    db:
        Open SQLAlchemy session (caller commits).
    user_id:
        Owner's user ID.
    name:
        Portfolio name.
    positions:
        Initial position list (at least one required).

    Returns
    -------
    Portfolio
        Flushed (not yet committed) Portfolio with populated positions.

    Raises
    ------
    PortfolioValidationError
        If name or positions fail validation.
    """
    _validate_name(name)
    _validate_positions(positions, require_nonempty=False)

    now = _now()
    portfolio = Portfolio(
        user_id=user_id,
        name=name.strip(),
        created_at=now,
        updated_at=now,
    )
    db.add(portfolio)
    db.flush()  # materialise portfolio.id before adding positions

    for pos in positions:
        db.add(
            PortfolioPosition(
                portfolio_id=portfolio.id,
                label=pos.label.strip(),
                weight_pct=pos.weight_pct,
                proxy_asset_id=pos.proxy_asset_id.strip(),
                created_at=now,
            )
        )
    db.flush()
    logger.info(
        "Portfolio created: id=%s name=%r user_id=%s positions=%d",
        portfolio.id, name, user_id, len(positions),
    )
    return portfolio


def update_portfolio_name(
    db: Session,
    portfolio_id: str,
    user_id: str,
    new_name: str,
) -> Portfolio:
    """Rename a portfolio.

    Raises
    ------
    PortfolioNotFoundError / PortfolioValidationError
    """
    _validate_name(new_name)
    portfolio = get_portfolio(db, portfolio_id, user_id)
    portfolio.name = new_name.strip()
    portfolio.updated_at = _now()
    db.flush()
    logger.info("Portfolio renamed: id=%s new_name=%r", portfolio_id, new_name)
    return portfolio


def set_positions(
    db: Session,
    portfolio_id: str,
    user_id: str,
    positions: list[PositionInput],
) -> Portfolio:
    """Atomically replace all positions in *portfolio_id*.

    Deletes every existing PortfolioPosition and inserts *positions* fresh.
    This is the canonical V1 update path — no partial position edits.

    Raises
    ------
    PortfolioNotFoundError / PortfolioValidationError
    """
    _validate_positions(positions)
    portfolio = get_portfolio(db, portfolio_id, user_id)

    # Delete existing positions via cascade-aware deletion
    for existing in list(portfolio.positions):
        db.delete(existing)
    db.flush()

    now = _now()
    for pos in positions:
        db.add(
            PortfolioPosition(
                portfolio_id=portfolio.id,
                label=pos.label.strip(),
                weight_pct=pos.weight_pct,
                proxy_asset_id=pos.proxy_asset_id.strip(),
                created_at=now,
            )
        )
    portfolio.updated_at = now
    db.flush()
    # Expire so that portfolio.positions is re-loaded from DB on next access
    db.expire(portfolio)
    logger.info(
        "Positions replaced: portfolio_id=%s positions=%d",
        portfolio_id, len(positions),
    )
    return portfolio


def delete_portfolio(db: Session, portfolio_id: str, user_id: str) -> None:
    """Delete *portfolio_id* and all its positions (cascade).

    Raises
    ------
    PortfolioNotFoundError
    """
    portfolio = get_portfolio(db, portfolio_id, user_id)
    db.delete(portfolio)
    db.flush()
    logger.info("Portfolio deleted: id=%s user_id=%s", portfolio_id, user_id)
