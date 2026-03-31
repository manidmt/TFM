'''
Tests for quant_risk.prod.auth.sessions

Uses SQLite in-memory via AuthDB — no real Postgres required.

Covers:
- create_session generates a token and persists the session
- get_session_by_token returns live session
- get_session_by_token returns None for expired session
- get_session_by_token returns None for unknown token
- renew_session pushes expires_at forward
- maybe_renew_session renews only when past threshold
- invalidate_session removes row and returns True
- invalidate_session returns False for unknown token
- invalidate_all_user_sessions removes all sessions for the user
- cascade delete: deleting a user removes all their sessions
- session token has expected format (URL-safe, non-empty)
- multiple sessions for same user are independent
'''

import pytest
from datetime import datetime, timedelta, timezone

import quant_risk.prod.auth.password as pw_mod
from quant_risk.prod.auth.db import AuthDB
from quant_risk.prod.auth.models import AppSession
from quant_risk.prod.auth.sessions import (
    SESSION_DURATION,
    RENEWAL_THRESHOLD,
    create_session,
    get_session_by_token,
    invalidate_all_user_sessions,
    invalidate_session,
    maybe_renew_session,
    renew_session,
)
from quant_risk.prod.auth.users import create_user

# Speed up tests
pw_mod.SCRYPT_N = 2


@pytest.fixture
def db():
    _db = AuthDB("sqlite:///:memory:")
    _db.create_tables()
    yield _db
    _db.close()


@pytest.fixture
def session(db):
    with db.session() as s:
        yield s


@pytest.fixture
def user(session):
    u = create_user(session, "alice@example.com", "password1")
    session.commit()
    return u


class TestCreateSession:
    def test_creates_row(self, session, user):
        sess = create_session(session, user)
        session.commit()
        assert sess.id is not None

    def test_token_is_nonempty(self, session, user):
        sess = create_session(session, user)
        session.commit()
        assert len(sess.token) > 20

    def test_token_is_url_safe(self, session, user):
        import re
        sess = create_session(session, user)
        session.commit()
        # URL-safe base64: alphanumeric + - _
        assert re.fullmatch(r"[A-Za-z0-9\-_]+", sess.token)

    def test_user_id_set(self, session, user):
        sess = create_session(session, user)
        session.commit()
        assert sess.user_id == user.id

    def test_expires_at_is_future(self, session, user):
        now = datetime.now(timezone.utc)
        sess = create_session(session, user)
        session.commit()
        assert sess.expires_at > now

    def test_expires_at_about_14_days(self, session, user):
        now = datetime.now(timezone.utc)
        sess = create_session(session, user)
        session.commit()
        delta = sess.expires_at.replace(tzinfo=timezone.utc) - now
        assert timedelta(days=13) < delta < timedelta(days=15)

    def test_two_tokens_are_different(self, session, user):
        s1 = create_session(session, user)
        s2 = create_session(session, user)
        session.commit()
        assert s1.token != s2.token


class TestGetSessionByToken:
    def test_returns_live_session(self, session, user):
        sess = create_session(session, user)
        session.commit()
        found = get_session_by_token(session, sess.token)
        assert found is not None
        assert found.id == sess.id

    def test_returns_none_for_unknown_token(self, session):
        assert get_session_by_token(session, "unknowntoken") is None

    def test_returns_none_for_expired_session(self, session, user):
        from sqlalchemy import select
        sess = create_session(session, user)
        session.commit()
        # Manually expire the session
        sess.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.commit()
        assert get_session_by_token(session, sess.token) is None

    def test_loads_user_relationship(self, session, user):
        sess = create_session(session, user)
        session.commit()
        found = get_session_by_token(session, sess.token)
        assert found.user.email == "alice@example.com"


class TestRenewSession:
    def test_renew_pushes_expires_at(self, session, user):
        sess = create_session(session, user)
        session.commit()
        old_expires = sess.expires_at
        import time; time.sleep(0.01)
        renewed = renew_session(session, sess)
        session.commit()
        assert renewed.expires_at >= old_expires

    def test_renew_updates_renewed_at(self, session, user):
        sess = create_session(session, user)
        session.commit()
        old_renewed = sess.renewed_at
        import time; time.sleep(0.01)
        renew_session(session, sess)
        session.commit()
        assert sess.renewed_at >= old_renewed

    def test_renewed_session_still_valid(self, session, user):
        sess = create_session(session, user)
        session.commit()
        renew_session(session, sess)
        session.commit()
        found = get_session_by_token(session, sess.token)
        assert found is not None


class TestMaybeRenewSession:
    def test_does_not_renew_when_fresh(self, session, user):
        sess = create_session(session, user)
        session.commit()
        old_expires = sess.expires_at
        maybe_renew_session(session, sess)
        session.commit()
        # expires_at should be unchanged (or differ by < 1s)
        assert abs((sess.expires_at.replace(tzinfo=timezone.utc) - old_expires.replace(tzinfo=timezone.utc)).total_seconds()) < 2

    def test_renews_when_past_threshold(self, session, user):
        sess = create_session(session, user)
        session.commit()
        # Simulate time passing beyond RENEWAL_THRESHOLD
        old_time = datetime.now(timezone.utc) - RENEWAL_THRESHOLD - timedelta(seconds=1)
        sess.renewed_at = old_time
        session.commit()
        maybe_renew_session(session, sess)
        session.commit()
        assert sess.renewed_at > old_time


class TestInvalidateSession:
    def test_invalidate_removes_session(self, session, user):
        sess = create_session(session, user)
        session.commit()
        result = invalidate_session(session, sess.token)
        session.commit()
        assert result is True
        assert get_session_by_token(session, sess.token) is None

    def test_invalidate_unknown_token_returns_false(self, session):
        result = invalidate_session(session, "nosuchtokenxyz")
        session.commit()
        assert result is False

    def test_invalidate_does_not_delete_other_sessions(self, session, user):
        s1 = create_session(session, user)
        s2 = create_session(session, user)
        session.commit()
        invalidate_session(session, s1.token)
        session.commit()
        assert get_session_by_token(session, s2.token) is not None


class TestInvalidateAllUserSessions:
    def test_deletes_all_sessions(self, session, user):
        create_session(session, user)
        create_session(session, user)
        create_session(session, user)
        session.commit()
        count = invalidate_all_user_sessions(session, user.id)
        session.commit()
        assert count == 3

    def test_does_not_delete_other_user_sessions(self, session, user):
        other = create_user(session, "bob@example.com", "password2")
        session.commit()
        create_session(session, other)
        session.commit()
        invalidate_all_user_sessions(session, user.id)
        session.commit()
        from sqlalchemy import select
        remaining = session.execute(
            select(AppSession).where(AppSession.user_id == other.id)
        ).scalars().all()
        assert len(remaining) == 1

    def test_returns_zero_when_no_sessions(self, session, user):
        count = invalidate_all_user_sessions(session, user.id)
        session.commit()
        assert count == 0


class TestCascadeDelete:
    def test_deleting_user_removes_sessions(self, session, user):
        from sqlalchemy import select
        create_session(session, user)
        create_session(session, user)
        session.commit()
        session.delete(user)
        session.commit()
        remaining = session.execute(
            select(AppSession).where(AppSession.user_id == user.id)
        ).scalars().all()
        assert remaining == []
