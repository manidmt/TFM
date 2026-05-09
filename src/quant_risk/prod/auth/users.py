'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: User CRUD operations for the auth layer (rpi5.md §14.2, §14.3).

All functions accept an open SQLAlchemy Session and return ORM objects.
Callers are responsible for committing / rolling back.

Account lifecycle (rpi5.md §14.2):
  - accounts are created manually by admin (no public sign-up)
  - initial bootstrap admin is created via CLI
  - new users receive temporary credentials; must_change_password=True
  - password reset by admin also sets must_change_password=True

Roles (rpi5.md §14.3):
  - ``user``  — can authenticate and manage own portfolios
  - ``admin`` — additionally has access to /ops routes
'''

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from quant_risk.prod.auth.models import VALID_ROLES, User
from quant_risk.prod.auth.password import hash_password

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UserAlreadyExistsError(ValueError):
    """Raised when creating a user with an e-mail that already exists."""


class InvalidRoleError(ValueError):
    """Raised when an invalid role string is supplied."""


class UserPendingApprovalError(Exception):
    """Raised by authenticate() when credentials are valid but account awaits admin approval."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_user_by_email(db: Session, email: str) -> User | None:
    """Return the User with *email* (case-insensitive), or None."""
    return db.execute(
        select(User).where(User.email == _normalize_email(email))
    ).scalar_one_or_none()


def get_user_by_id(db: Session, user_id: str) -> User | None:
    """Return the User with *user_id*, or None."""
    return db.get(User, user_id)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def create_user(
    db: Session,
    email: str,
    plain_password: str,
    role: str = "user",
    must_change_password: bool = True,
    is_approved: bool = True,
) -> User:
    """Create a new user account.

    Parameters
    ----------
    db:
        Open SQLAlchemy session (caller commits).
    email:
        Login e-mail (normalised to lowercase).
    plain_password:
        Temporary plaintext password.  Must meet ``MIN_PASSWORD_LEN``.
    role:
        ``'user'`` or ``'admin'``.
    must_change_password:
        If True (default), the user must change their password at first login.

    Returns
    -------
    User
        The newly created and flushed (not yet committed) User object.

    Raises
    ------
    UserAlreadyExistsError
        If a user with this e-mail already exists.
    InvalidRoleError
        If *role* is not a valid role string.
    ValueError
        If *plain_password* does not meet length requirements.
    """
    if role not in VALID_ROLES:
        raise InvalidRoleError(
            f"Invalid role '{role}'. Valid roles: {sorted(VALID_ROLES)}"
        )

    norm_email = _normalize_email(email)
    now = _now()

    user = User(
        email=norm_email,
        password_hash=hash_password(plain_password),
        role=role,
        must_change_password=must_change_password,
        is_active=True,
        is_approved=is_approved,
        created_at=now,
        updated_at=now,
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise UserAlreadyExistsError(
            f"A user with email '{norm_email}' already exists."
        )

    logger.info("Created user email=%s role=%s", norm_email, role)
    return user


def authenticate(db: Session, email: str, plain_password: str) -> User | None:
    """Return the User if credentials are valid, or None.

    Returns None for unknown e-mail, wrong password, or inactive account.
    Raises UserPendingApprovalError when credentials are correct but the
    account has not been approved by an admin yet.

    Callers should check ``user.must_change_password`` after a successful
    authentication and redirect accordingly.
    """
    from quant_risk.prod.auth.password import verify_password

    user = get_user_by_email(db, email)
    if user is None or not user.is_active:
        return None
    if not verify_password(plain_password, user.password_hash):
        logger.debug("Failed login attempt for email=%s", _normalize_email(email))
        return None
    if not user.is_approved:
        raise UserPendingApprovalError(user.email)
    return user


def change_password(
    db: Session,
    user: User,
    new_plain_password: str,
    *,
    clear_must_change: bool = True,
) -> User:
    """Update *user*'s password.

    Parameters
    ----------
    db:
        Open SQLAlchemy session (caller commits).
    user:
        The User to update.
    new_plain_password:
        New plaintext password.
    clear_must_change:
        If True (default), sets ``must_change_password=False`` after the change.

    Returns
    -------
    User
        The updated User object (not yet committed).
    """
    user.password_hash = hash_password(new_plain_password)
    if clear_must_change:
        user.must_change_password = False
    user.updated_at = _now()
    db.flush()
    logger.info("Password changed for user_id=%s", user.id)
    return user


def approve_user(db: Session, user: User) -> User:
    """Grant access to a pending user (set is_approved=True, is_active=True)."""
    user.is_approved = True
    user.is_active = True
    user.updated_at = _now()
    db.flush()
    logger.info("User approved email=%s user_id=%s", user.email, user.id)
    return user


def reject_user(db: Session, user: User) -> User:
    """Reject a pending signup (is_approved stays False, is_active=False)."""
    user.is_approved = False
    user.is_active = False
    user.updated_at = _now()
    db.flush()
    logger.info("User rejected email=%s user_id=%s", user.email, user.id)
    return user


def set_active(db: Session, user: User, active: bool) -> User:
    """Activate or deactivate *user*.

    Deactivating a user does not delete their sessions; callers should
    call ``invalidate_all_user_sessions`` if immediate session termination
    is required.
    """
    user.is_active = active
    user.updated_at = _now()
    db.flush()
    logger.info(
        "User %s set to active=%s (user_id=%s)", user.email, active, user.id
    )
    return user
