'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Auth and session management package (rpi5.md §14).

Public re-exports for convenience:
  from quant_risk.prod.auth import AuthDB, User, AppSession
  from quant_risk.prod.auth import hash_password, verify_password
  from quant_risk.prod.auth import create_user, authenticate
  from quant_risk.prod.auth import create_session, get_session_by_token
'''

from quant_risk.prod.auth.db import AuthDB
from quant_risk.prod.auth.models import AppSession, Portfolio, PortfolioPosition, User
from quant_risk.prod.auth.password import hash_password, verify_password
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
from quant_risk.prod.auth.sessions import (
    SESSION_DURATION,
    create_session,
    get_session_by_token,
    invalidate_all_user_sessions,
    invalidate_session,
    renew_session,
)
from quant_risk.prod.auth.users import (
    UserAlreadyExistsError,
    authenticate,
    change_password,
    create_user,
    get_user_by_email,
    get_user_by_id,
    set_active,
)

__all__ = [
    # DB
    "AuthDB",
    # Models
    "User",
    "AppSession",
    "Portfolio",
    "PortfolioPosition",
    # Password
    "hash_password",
    "verify_password",
    # Users
    "UserAlreadyExistsError",
    "create_user",
    "authenticate",
    "get_user_by_email",
    "get_user_by_id",
    "change_password",
    "set_active",
    # Sessions
    "SESSION_DURATION",
    "create_session",
    "get_session_by_token",
    "invalidate_session",
    "renew_session",
    "invalidate_all_user_sessions",
    # Portfolios
    "PortfolioNotFoundError",
    "PortfolioValidationError",
    "PositionInput",
    "create_portfolio",
    "get_portfolio",
    "list_user_portfolios",
    "update_portfolio_name",
    "set_positions",
    "delete_portfolio",
]
