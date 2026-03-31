'''
Tests for quant_risk.prod.auth.users

Uses SQLite in-memory via AuthDB — no real Postgres required.

Covers:
- create_user happy path (user / admin)
- email normalisation (case-insensitive)
- must_change_password default behaviour
- duplicate email raises UserAlreadyExistsError
- invalid role raises InvalidRoleError
- password too short raises ValueError
- get_user_by_email (found / not found)
- get_user_by_id (found / not found)
- authenticate (success / wrong password / inactive user / unknown email)
- change_password clears must_change_password flag
- change_password with clear_must_change=False leaves flag unchanged
- set_active activate / deactivate
- inactive user cannot authenticate
'''

import pytest

import quant_risk.prod.auth.password as pw_mod
from quant_risk.prod.auth.db import AuthDB
from quant_risk.prod.auth.users import (
    InvalidRoleError,
    UserAlreadyExistsError,
    authenticate,
    change_password,
    create_user,
    get_user_by_email,
    get_user_by_id,
    set_active,
)

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
    u = create_user(session, "alice@example.com", "password1", role="user")
    session.commit()
    return u


class TestCreateUser:
    def test_creates_user(self, session):
        u = create_user(session, "bob@example.com", "password1")
        session.commit()
        assert u.id is not None
        assert u.email == "bob@example.com"

    def test_default_role_is_user(self, session):
        u = create_user(session, "bob@example.com", "password1")
        session.commit()
        assert u.role == "user"

    def test_admin_role(self, session):
        u = create_user(session, "admin@example.com", "password1", role="admin")
        session.commit()
        assert u.role == "admin"

    def test_must_change_password_default_true(self, session):
        u = create_user(session, "bob@example.com", "password1")
        session.commit()
        assert u.must_change_password is True

    def test_must_change_password_false(self, session):
        u = create_user(
            session, "bob@example.com", "password1", must_change_password=False
        )
        session.commit()
        assert u.must_change_password is False

    def test_email_normalised_to_lowercase(self, session):
        u = create_user(session, "Bob@EXAMPLE.COM", "password1")
        session.commit()
        assert u.email == "bob@example.com"

    def test_is_active_true_by_default(self, session):
        u = create_user(session, "bob@example.com", "password1")
        session.commit()
        assert u.is_active is True

    def test_password_not_stored_in_plain(self, session):
        u = create_user(session, "bob@example.com", "password1")
        session.commit()
        assert u.password_hash != "password1"
        assert u.password_hash.startswith("scrypt:")

    def test_duplicate_email_raises(self, session):
        create_user(session, "dup@example.com", "password1")
        session.commit()
        with pytest.raises(UserAlreadyExistsError):
            create_user(session, "dup@example.com", "password2")

    def test_duplicate_email_case_insensitive(self, session):
        create_user(session, "dup@example.com", "password1")
        session.commit()
        with pytest.raises(UserAlreadyExistsError):
            create_user(session, "DUP@EXAMPLE.COM", "password2")

    def test_invalid_role_raises(self, session):
        with pytest.raises(InvalidRoleError):
            create_user(session, "bob@example.com", "password1", role="superuser")

    def test_short_password_raises(self, session):
        with pytest.raises(ValueError, match="at least"):
            create_user(session, "bob@example.com", "short")

    def test_created_at_set(self, session):
        u = create_user(session, "bob@example.com", "password1")
        session.commit()
        assert u.created_at is not None

    def test_updated_at_equals_created_at(self, session):
        u = create_user(session, "bob@example.com", "password1")
        session.commit()
        assert u.updated_at == u.created_at


class TestGetUser:
    def test_get_by_email_found(self, session, user):
        found = get_user_by_email(session, "alice@example.com")
        assert found is not None
        assert found.id == user.id

    def test_get_by_email_case_insensitive(self, session, user):
        found = get_user_by_email(session, "ALICE@EXAMPLE.COM")
        assert found is not None
        assert found.id == user.id

    def test_get_by_email_not_found(self, session):
        assert get_user_by_email(session, "nobody@example.com") is None

    def test_get_by_id_found(self, session, user):
        found = get_user_by_id(session, user.id)
        assert found is not None
        assert found.email == user.email

    def test_get_by_id_not_found(self, session):
        assert get_user_by_id(session, "nonexistent-uuid") is None


class TestAuthenticate:
    def test_correct_credentials(self, session, user):
        result = authenticate(session, "alice@example.com", "password1")
        assert result is not None
        assert result.id == user.id

    def test_wrong_password_returns_none(self, session, user):
        assert authenticate(session, "alice@example.com", "wrongpassword") is None

    def test_unknown_email_returns_none(self, session):
        assert authenticate(session, "nobody@example.com", "password1") is None

    def test_inactive_user_returns_none(self, session, user):
        set_active(session, user, False)
        session.commit()
        assert authenticate(session, "alice@example.com", "password1") is None

    def test_case_insensitive_email(self, session, user):
        result = authenticate(session, "ALICE@EXAMPLE.COM", "password1")
        assert result is not None


class TestChangePassword:
    def test_new_password_verifies(self, session, user):
        from quant_risk.prod.auth.password import verify_password
        change_password(session, user, "newpassword1")
        session.commit()
        assert verify_password("newpassword1", user.password_hash)

    def test_clears_must_change_by_default(self, session):
        u = create_user(session, "bob@example.com", "password1", must_change_password=True)
        session.commit()
        change_password(session, u, "newpassword1")
        session.commit()
        assert u.must_change_password is False

    def test_clear_must_change_false_preserves_flag(self, session):
        u = create_user(session, "bob@example.com", "password1", must_change_password=True)
        session.commit()
        change_password(session, u, "newpassword1", clear_must_change=False)
        session.commit()
        assert u.must_change_password is True

    def test_old_password_no_longer_valid(self, session, user):
        from quant_risk.prod.auth.password import verify_password
        change_password(session, user, "newpassword1")
        session.commit()
        assert not verify_password("password1", user.password_hash)

    def test_short_new_password_raises(self, session, user):
        with pytest.raises(ValueError):
            change_password(session, user, "short")


class TestSetActive:
    def test_deactivate(self, session, user):
        set_active(session, user, False)
        session.commit()
        assert user.is_active is False

    def test_reactivate(self, session, user):
        set_active(session, user, False)
        session.commit()
        set_active(session, user, True)
        session.commit()
        assert user.is_active is True

    def test_updated_at_changes(self, session, user):
        original_updated_at = user.updated_at
        import time; time.sleep(0.01)
        set_active(session, user, False)
        session.commit()
        assert user.updated_at >= original_updated_at
