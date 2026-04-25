'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Auth endpoints — login, logout, session introspection, password change
              (rpi5.md §14).

Routes
------
POST /api/auth/login
    Authenticate with email + password.
    Sets an HttpOnly session cookie on success.
    Returns 401 on bad credentials.

POST /api/auth/logout
    Invalidate the current session and clear the cookie.

GET /api/auth/me
    Return the current user's profile.
    Includes must_change_password so the frontend can redirect.

POST /api/auth/change-password
    Change the authenticated user's password.
    Invalidates all existing sessions after a successful change,
    then issues a new session so the user stays logged in.

Cookie policy (rpi5.md §14.4)
------------------------------
- HttpOnly: JavaScript cannot read the token.
- Secure: only sent over HTTPS (configurable per environment).
- SameSite=lax: CSRF protection.
- Expiry: 14 days, renewed on activity by get_current_user.
'''

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from quant_risk.prod.api.config import AppConfig
from quant_risk.prod.api.deps import get_auth_db, get_config, get_current_user
from quant_risk.prod.auth.models import User
from quant_risk.prod.auth.notify import notify_admin_signup
from quant_risk.prod.auth.sessions import (
    SESSION_DURATION,
    create_session,
    invalidate_all_user_sessions,
    invalidate_session,
)
from quant_risk.prod.auth.users import (
    UserAlreadyExistsError,
    UserPendingApprovalError,
    authenticate,
    change_password,
    create_user,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: str
    password: str


class SignupRequest(BaseModel):
    email: str
    password: str


class SignupResponse(BaseModel):
    message: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class UserOut(BaseModel):
    user_id: str
    email: str
    role: str
    must_change_password: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_session_cookie(response: Response, token: str, config: AppConfig) -> None:
    response.set_cookie(
        key=config.cookie_name,
        value=token,
        max_age=int(SESSION_DURATION.total_seconds()),
        httponly=config.cookie_httponly,
        secure=config.cookie_secure,
        samesite=config.cookie_samesite,
        path="/",
    )


def _clear_session_cookie(response: Response, config: AppConfig) -> None:
    response.delete_cookie(
        key=config.cookie_name,
        httponly=config.cookie_httponly,
        secure=config.cookie_secure,
        samesite=config.cookie_samesite,
        path="/",
    )


def _user_out(user: User) -> UserOut:
    return UserOut(
        user_id=user.id,
        email=user.email,
        role=user.role,
        must_change_password=user.must_change_password,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
def signup(
    body: SignupRequest,
    config: AppConfig = Depends(get_config),
    db: Session = Depends(get_auth_db),
):
    """Public registration endpoint.

    Creates an account with is_approved=False. The admin must approve it
    before the user can log in. Fires a notification email if SMTP is configured.
    """
    try:
        create_user(
            db,
            email=body.email,
            plain_password=body.password,
            role="user",
            must_change_password=False,
            is_approved=False,
        )
        db.commit()
    except UserAlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    notify_admin_signup(body.email, config)
    return SignupResponse(message="Account created. Awaiting admin approval.")


@router.post("/login", response_model=UserOut)
def login(
    body: LoginRequest,
    response: Response,
    config: AppConfig = Depends(get_config),
    db: Session = Depends(get_auth_db),
):
    """Authenticate and set a session cookie.

    Returns HTTP 403 with detail "pending_approval" if the account exists but
    has not been approved by an admin yet.
    The caller should check ``must_change_password`` in the response and
    redirect to the password-change page if True.
    """
    try:
        user = authenticate(db, body.email, body.password)
    except UserPendingApprovalError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="pending_approval",
        )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials.",
        )
    sess = create_session(db, user)
    db.commit()
    _set_session_cookie(response, sess.token, config)
    return _user_out(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    request: Request,
    response: Response,
    config: AppConfig = Depends(get_config),
    db: Session = Depends(get_auth_db),
    _user: User = Depends(get_current_user),
):
    """Invalidate the current session and clear the cookie."""
    token: str | None = request.cookies.get(config.cookie_name)
    if token:
        invalidate_session(db, token)
        db.commit()
    _clear_session_cookie(response, config)


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    """Return the current user's profile."""
    return _user_out(current_user)


@router.post("/change-password", response_model=UserOut)
def change_password_endpoint(
    body: ChangePasswordRequest,
    request: Request,
    response: Response,
    config: AppConfig = Depends(get_config),
    db: Session = Depends(get_auth_db),
    current_user: User = Depends(get_current_user),
):
    """Change the authenticated user's password.

    - Verifies the current password before accepting the change.
    - Invalidates all existing sessions.
    - Issues a new session so the user stays logged in.
    """
    from quant_risk.prod.auth.password import verify_password

    if not verify_password(body.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )

    try:
        change_password(db, current_user, body.new_password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # Invalidate all existing sessions (including this one).
    invalidate_all_user_sessions(db, current_user.id)

    # Re-issue a fresh session so the user stays logged in.
    new_sess = create_session(db, current_user)
    db.commit()

    _set_session_cookie(response, new_sess.token, config)
    return _user_out(current_user)
