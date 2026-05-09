'''
Tests for quant_risk.prod.auth.cli

Uses SQLite in-memory.  CLI functions are called directly (not via subprocess)
so we can inject the --db-url argument and inspect the database after each call.

Covers:
- create-admin creates an admin user
- create-admin sets must_change_password=True
- create-admin with duplicate email prints error and exits 1
- create-admin with short password exits 1
- create-user creates a regular user with role=user
- create-user sets must_change_password=True
- reset-password updates the password and forces re-change
- reset-password for unknown email exits 1
- reset-password with short password exits 1
- list-users prints user table
- list-users prints "No users found" when empty
'''

import sys
import pytest

import quant_risk.prod.auth.password as pw_mod
from quant_risk.prod.auth.cli import main
from quant_risk.prod.auth.db import AuthDB
from quant_risk.prod.auth.password import verify_password
from quant_risk.prod.auth.users import get_user_by_email

# Speed up tests
pw_mod.SCRYPT_N = 2


@pytest.fixture
def db_url(tmp_path):
    """Return a file-based SQLite URL so multiple CLI calls share the same DB."""
    return f"sqlite:///{tmp_path / 'test_auth.db'}"


def _run(*args: str) -> None:
    """Invoke the CLI main() with the given args (raises SystemExit on error)."""
    main(list(args))


class TestCreateAdmin:
    def test_creates_admin_user(self, db_url, capsys):
        _run("--db-url", db_url, "create-admin",
             "--email", "admin@example.com", "--password", "adminpass")
        db = AuthDB(db_url)
        with db.session() as s:
            user = get_user_by_email(s, "admin@example.com")
        db.close()
        assert user is not None
        assert user.role == "admin"

    def test_must_change_password_true(self, db_url):
        _run("--db-url", db_url, "create-admin",
             "--email", "admin@example.com", "--password", "adminpass")
        db = AuthDB(db_url)
        with db.session() as s:
            user = get_user_by_email(s, "admin@example.com")
        db.close()
        assert user.must_change_password is True

    def test_output_includes_email(self, db_url, capsys):
        _run("--db-url", db_url, "create-admin",
             "--email", "admin@example.com", "--password", "adminpass")
        out = capsys.readouterr().out
        assert "admin@example.com" in out

    def test_duplicate_email_exits_1(self, db_url):
        _run("--db-url", db_url, "create-admin",
             "--email", "admin@example.com", "--password", "adminpass")
        with pytest.raises(SystemExit) as exc_info:
            _run("--db-url", db_url, "create-admin",
                 "--email", "admin@example.com", "--password", "adminpass2")
        assert exc_info.value.code == 1

    def test_short_password_exits_1(self, db_url):
        with pytest.raises(SystemExit) as exc_info:
            _run("--db-url", db_url, "create-admin",
                 "--email", "admin@example.com", "--password", "short")
        assert exc_info.value.code == 1


class TestCreateUser:
    def test_creates_user_with_role_user(self, db_url):
        _run("--db-url", db_url, "create-user",
             "--email", "user@example.com", "--password", "userpass1")
        db = AuthDB(db_url)
        with db.session() as s:
            user = get_user_by_email(s, "user@example.com")
        db.close()
        assert user is not None
        assert user.role == "user"

    def test_must_change_password_true(self, db_url):
        _run("--db-url", db_url, "create-user",
             "--email", "user@example.com", "--password", "userpass1")
        db = AuthDB(db_url)
        with db.session() as s:
            user = get_user_by_email(s, "user@example.com")
        db.close()
        assert user.must_change_password is True

    def test_duplicate_email_exits_1(self, db_url):
        _run("--db-url", db_url, "create-user",
             "--email", "user@example.com", "--password", "userpass1")
        with pytest.raises(SystemExit) as exc_info:
            _run("--db-url", db_url, "create-user",
                 "--email", "user@example.com", "--password", "userpass2")
        assert exc_info.value.code == 1

    def test_short_password_exits_1(self, db_url):
        with pytest.raises(SystemExit) as exc_info:
            _run("--db-url", db_url, "create-user",
                 "--email", "user@example.com", "--password", "short")
        assert exc_info.value.code == 1


class TestResetPassword:
    def test_updates_password(self, db_url):
        _run("--db-url", db_url, "create-user",
             "--email", "user@example.com", "--password", "oldpasswd")
        _run("--db-url", db_url, "reset-password",
             "--email", "user@example.com", "--password", "newpasswd1")
        db = AuthDB(db_url)
        with db.session() as s:
            user = get_user_by_email(s, "user@example.com")
            ph = user.password_hash
        db.close()
        assert verify_password("newpasswd1", ph)

    def test_sets_must_change_password(self, db_url):
        _run("--db-url", db_url, "create-user",
             "--email", "user@example.com", "--password", "oldpasswd")
        # Simulate user cleared the flag
        db = AuthDB(db_url)
        with db.session() as s:
            from quant_risk.prod.auth.users import change_password
            user = get_user_by_email(s, "user@example.com")
            change_password(s, user, "oldpasswd")  # clears must_change
            s.commit()
        db.close()
        # Now reset
        _run("--db-url", db_url, "reset-password",
             "--email", "user@example.com", "--password", "newpasswd1")
        db = AuthDB(db_url)
        with db.session() as s:
            user = get_user_by_email(s, "user@example.com")
            mcp = user.must_change_password
        db.close()
        assert mcp is True

    def test_unknown_email_exits_1(self, db_url):
        # Need at least one user so tables exist
        _run("--db-url", db_url, "create-user",
             "--email", "user@example.com", "--password", "oldpasswd")
        with pytest.raises(SystemExit) as exc_info:
            _run("--db-url", db_url, "reset-password",
                 "--email", "nobody@example.com", "--password", "newpasswd1")
        assert exc_info.value.code == 1

    def test_short_new_password_exits_1(self, db_url):
        _run("--db-url", db_url, "create-user",
             "--email", "user@example.com", "--password", "oldpasswd")
        with pytest.raises(SystemExit) as exc_info:
            _run("--db-url", db_url, "reset-password",
                 "--email", "user@example.com", "--password", "short")
        assert exc_info.value.code == 1


class TestListUsers:
    def test_no_users_message(self, db_url, capsys):
        # Create tables but no users
        AuthDB(db_url).create_tables()
        _run("--db-url", db_url, "list-users")
        out = capsys.readouterr().out
        assert "No users found" in out

    def test_shows_created_users(self, db_url, capsys):
        _run("--db-url", db_url, "create-user",
             "--email", "alice@example.com", "--password", "password1")
        _run("--db-url", db_url, "create-admin",
             "--email", "admin@example.com", "--password", "adminpass")
        _run("--db-url", db_url, "list-users")
        out = capsys.readouterr().out
        assert "alice@example.com" in out
        assert "admin@example.com" in out

    def test_shows_role_column(self, db_url, capsys):
        _run("--db-url", db_url, "create-admin",
             "--email", "admin@example.com", "--password", "adminpass")
        _run("--db-url", db_url, "list-users")
        out = capsys.readouterr().out
        assert "admin" in out
