'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: CLI for user/account management (rpi5.md §14.2).

Commands
--------
    python -m quant_risk.prod.auth.cli create-admin
        --email admin@example.com --password "TmpP@ss1"
        [--db-url postgresql://user:pass@host/db]

    python -m quant_risk.prod.auth.cli create-user
        --email user@example.com --password "TmpP@ss1"
        [--db-url ...]

    python -m quant_risk.prod.auth.cli reset-password
        --email user@example.com --password "NewP@ss1"
        [--db-url ...]

    python -m quant_risk.prod.auth.cli list-users
        [--db-url ...]

Environment variable: QUANT_RISK_POSTGRES_URL is used when --db-url is absent.

Account lifecycle notes (rpi5.md §14.2):
  - No public self-registration; all accounts are created via this CLI.
  - create-admin / create-user always set must_change_password=True.
  - reset-password sets must_change_password=True so the user must re-change
    at next login.
'''

from __future__ import annotations

import argparse
import sys

from quant_risk.prod.auth.db import AuthDB
from quant_risk.prod.auth.users import (
    InvalidRoleError,
    UserAlreadyExistsError,
    change_password,
    create_user,
    get_user_by_email,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_db(db_url: str | None) -> AuthDB:
    db = AuthDB(db_url)
    db.create_tables()
    return db


def _die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_create_admin(args: argparse.Namespace) -> None:
    db = _get_db(args.db_url)
    try:
        with db.session() as s:
            user = create_user(
                s,
                email=args.email,
                plain_password=args.password,
                role="admin",
                must_change_password=True,
            )
            s.commit()
        print(f"Admin created: {user.email} (id={user.id})")
        print("Note: user must change password at first login.")
    except UserAlreadyExistsError as exc:
        _die(str(exc))
    except ValueError as exc:
        _die(str(exc))
    finally:
        db.close()


def cmd_create_user(args: argparse.Namespace) -> None:
    db = _get_db(args.db_url)
    try:
        with db.session() as s:
            user = create_user(
                s,
                email=args.email,
                plain_password=args.password,
                role="user",
                must_change_password=True,
            )
            s.commit()
        print(f"User created: {user.email} (id={user.id})")
        print("Note: user must change password at first login.")
    except UserAlreadyExistsError as exc:
        _die(str(exc))
    except ValueError as exc:
        _die(str(exc))
    finally:
        db.close()


def cmd_reset_password(args: argparse.Namespace) -> None:
    db = _get_db(args.db_url)
    try:
        with db.session() as s:
            user = get_user_by_email(s, args.email)
            if user is None:
                _die(f"No user found with email '{args.email}'.")
            change_password(
                s, user, args.password, clear_must_change=False
            )
            # Reset always forces re-change.
            user.must_change_password = True
            s.commit()
        print(f"Password reset for {args.email}.")
        print("Note: user must change password at next login.")
    except ValueError as exc:
        _die(str(exc))
    finally:
        db.close()


def cmd_list_users(args: argparse.Namespace) -> None:
    from sqlalchemy import select
    from quant_risk.prod.auth.models import User

    db = _get_db(args.db_url)
    try:
        with db.session() as s:
            users = s.execute(select(User).order_by(User.email)).scalars().all()
        if not users:
            print("No users found.")
            return
        print(f"{'EMAIL':<40} {'ROLE':<8} {'ACTIVE':<7} {'MUST_CHANGE'}")
        print("-" * 72)
        for u in users:
            print(
                f"{u.email:<40} {u.role:<8} {str(u.is_active):<7} "
                f"{u.must_change_password}"
            )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quant-risk-auth",
        description="quant-risk user/account management CLI (rpi5.md §14.2)",
    )
    p.add_argument(
        "--db-url",
        default=None,
        help=(
            "SQLAlchemy DB URL (e.g. postgresql://user:pass@host/db). "
            "Falls back to QUANT_RISK_POSTGRES_URL environment variable."
        ),
    )

    sub = p.add_subparsers(dest="command", required=True)

    # create-admin
    ca = sub.add_parser("create-admin", help="Bootstrap an admin account.")
    ca.add_argument("--email", required=True)
    ca.add_argument("--password", required=True)

    # create-user
    cu = sub.add_parser("create-user", help="Create a regular user account.")
    cu.add_argument("--email", required=True)
    cu.add_argument("--password", required=True)

    # reset-password
    rp = sub.add_parser("reset-password", help="Reset a user's password (sets must_change_password=True).")
    rp.add_argument("--email", required=True)
    rp.add_argument("--password", required=True)

    # list-users
    sub.add_parser("list-users", help="List all user accounts.")

    return p


_HANDLERS = {
    "create-admin": cmd_create_admin,
    "create-user": cmd_create_user,
    "reset-password": cmd_reset_password,
    "list-users": cmd_list_users,
}


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _HANDLERS[args.command](args)


if __name__ == "__main__":
    main()
