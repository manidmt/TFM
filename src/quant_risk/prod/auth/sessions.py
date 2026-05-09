'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Session CRUD for the auth layer (rpi5.md §14.4).

Session policy (rpi5.md §14.4):
  - cookie-based session token (issued by FastAPI, stored in HttpOnly cookie)
  - session record persisted in Postgres
  - valid for SESSION_DURATION (14 days)
  - renewed on activity: if the session is older than RENEWAL_THRESHOLD,
    expires_at is pushed forward by SESSION_DURATION from now

Token generation
----------------
``secrets.token_urlsafe(32)`` produces a 43-character URL-safe string with
~192 bits of entropy — sufficient for a session token.
'''

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from quant_risk.prod.auth.models import AppSession, User

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

SESSION_DURATION = timedelta(days=14)

# Renew the session when more than half the lifetime has elapsed.
RENEWAL_THRESHOLD = SESSION_DURATION / 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create_session(db: Session, user: User) -> AppSession:
    """Create and persist a new session for *user*.

    Parameters
    ----------
    db:
        Open SQLAlchemy session (caller commits).
    user:
        The authenticated User.

    Returns
    -------
    AppSession
        The newly created session (flushed, not committed).
        ``session.token`` is the value to store in the client cookie.
    """
    now = _now()
    token = secrets.token_urlsafe(32)
    sess = AppSession(
        user_id=user.id,
        token=token,
        created_at=now,
        expires_at=now + SESSION_DURATION,
        renewed_at=now,
    )
    db.add(sess)
    db.flush()
    logger.info("Session created for user_id=%s", user.id)
    return sess


def get_session_by_token(db: Session, token: str) -> AppSession | None:
    """Return the live AppSession for *token*, or None if absent/expired.

    A session is live if ``expires_at > now``.
    """
    now = _now()
    return db.execute(
        select(AppSession).where(
            AppSession.token == token,
            AppSession.expires_at > now,
        )
    ).scalar_one_or_none()


def renew_session(db: Session, sess: AppSession) -> AppSession:
    """Push *sess* expiry forward by SESSION_DURATION from now.

    Should be called when the session has passed RENEWAL_THRESHOLD.
    Callers that want auto-renewal on every request can call this
    unconditionally — it is cheap.

    Returns the updated AppSession (flushed, not committed).
    """
    now = _now()
    sess.expires_at = now + SESSION_DURATION
    sess.renewed_at = now
    db.flush()
    return sess


def maybe_renew_session(db: Session, sess: AppSession) -> AppSession:
    """Renew *sess* only if more than RENEWAL_THRESHOLD has elapsed.

    Returns the (possibly updated) AppSession.
    """
    now = _now()
    elapsed = now - sess.renewed_at.replace(tzinfo=timezone.utc) if sess.renewed_at.tzinfo is None else now - sess.renewed_at
    if elapsed >= RENEWAL_THRESHOLD:
        return renew_session(db, sess)
    return sess


def invalidate_session(db: Session, token: str) -> bool:
    """Delete the session with *token* (logout).

    Returns True if a session was deleted, False if no matching session found.
    """
    result = db.execute(
        delete(AppSession).where(AppSession.token == token)
    )
    db.flush()
    deleted = result.rowcount > 0
    if deleted:
        logger.info("Session invalidated (token prefix: %s...)", token[:8])
    return deleted


def invalidate_all_user_sessions(db: Session, user_id: str) -> int:
    """Delete all sessions for *user_id*.

    Returns the number of deleted sessions.
    Useful after password change or deactivation.
    """
    result = db.execute(
        delete(AppSession).where(AppSession.user_id == user_id)
    )
    db.flush()
    count = result.rowcount
    logger.info("Invalidated %d session(s) for user_id=%s", count, user_id)
    return count
