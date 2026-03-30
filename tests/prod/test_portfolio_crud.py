'''
Tests for quant_risk.prod.auth.portfolios (CRUD) and Portfolio/PortfolioPosition models.

Uses SQLite in-memory via AuthDB — no real Postgres required.

Covers:
- create_portfolio: happy path, positions stored, validation errors
- get_portfolio: found / not found / wrong user
- list_user_portfolios: ordering, isolation between users
- update_portfolio_name: rename, validation
- set_positions: atomic replacement, validation
- delete_portfolio: cascade deletes positions, not-found
- cascade delete: deleting a user removes portfolios and positions
- validation: empty name, empty label, negative weight, empty proxy
- PositionInput weight = 0 is allowed (zero-weight position)
'''

import pytest

import quant_risk.prod.auth.password as pw_mod
from quant_risk.prod.auth.db import AuthDB
from quant_risk.prod.auth.models import Portfolio, PortfolioPosition
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
from quant_risk.prod.auth.users import create_user

# Speed up scrypt
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


@pytest.fixture
def other_user(session):
    u = create_user(session, "bob@example.com", "password2")
    session.commit()
    return u


_POS = [
    PositionInput(label="AAPL", weight_pct=50.0, proxy_asset_id="us_equities"),
    PositionInput(label="BTC",  weight_pct=50.0, proxy_asset_id="btc"),
]


class TestCreatePortfolio:
    def test_creates_portfolio(self, session, user):
        p = create_portfolio(session, user.id, "My Portfolio", _POS)
        session.commit()
        assert p.id is not None
        assert p.name == "My Portfolio"
        assert p.user_id == user.id

    def test_positions_stored(self, session, user):
        p = create_portfolio(session, user.id, "My Portfolio", _POS)
        session.commit()
        session.expire(p)
        loaded = session.get(Portfolio, p.id)
        assert len(loaded.positions) == 2
        labels = {pos.label for pos in loaded.positions}
        assert labels == {"AAPL", "BTC"}

    def test_position_fields(self, session, user):
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        session.expire(p)
        loaded = session.get(Portfolio, p.id)
        aapl = next(pos for pos in loaded.positions if pos.label == "AAPL")
        assert aapl.weight_pct == 50.0
        assert aapl.proxy_asset_id == "us_equities"

    def test_created_at_set(self, session, user):
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        assert p.created_at is not None
        assert p.updated_at is not None

    def test_name_stripped(self, session, user):
        p = create_portfolio(session, user.id, "  My Portfolio  ", _POS)
        session.commit()
        assert p.name == "My Portfolio"

    def test_empty_name_raises(self, session, user):
        with pytest.raises(PortfolioValidationError, match="name"):
            create_portfolio(session, user.id, "", _POS)

    def test_whitespace_name_raises(self, session, user):
        with pytest.raises(PortfolioValidationError, match="name"):
            create_portfolio(session, user.id, "   ", _POS)

    def test_empty_positions_allowed(self, session, user):
        # Creating a portfolio with no positions is valid; positions are added later.
        p = create_portfolio(session, user.id, "P", [])
        session.commit()
        assert p.id is not None
        assert p.positions == []

    def test_negative_weight_raises(self, session, user):
        bad = [PositionInput(label="AAPL", weight_pct=-10.0, proxy_asset_id="us_equities")]
        with pytest.raises(PortfolioValidationError, match="weight_pct"):
            create_portfolio(session, user.id, "P", bad)

    def test_zero_weight_allowed(self, session, user):
        zero = [PositionInput(label="AAPL", weight_pct=0.0, proxy_asset_id="us_equities")]
        p = create_portfolio(session, user.id, "P", zero)
        session.commit()
        assert p.id is not None

    def test_empty_label_raises(self, session, user):
        bad = [PositionInput(label="", weight_pct=50.0, proxy_asset_id="us_equities")]
        with pytest.raises(PortfolioValidationError, match="label"):
            create_portfolio(session, user.id, "P", bad)

    def test_empty_proxy_raises(self, session, user):
        bad = [PositionInput(label="AAPL", weight_pct=50.0, proxy_asset_id="")]
        with pytest.raises(PortfolioValidationError, match="proxy_asset_id"):
            create_portfolio(session, user.id, "P", bad)


class TestGetPortfolio:
    def test_found(self, session, user):
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        found = get_portfolio(session, p.id, user.id)
        assert found.id == p.id

    def test_wrong_user_raises(self, session, user, other_user):
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        with pytest.raises(PortfolioNotFoundError):
            get_portfolio(session, p.id, other_user.id)

    def test_nonexistent_raises(self, session, user):
        with pytest.raises(PortfolioNotFoundError):
            get_portfolio(session, "nonexistent-uuid", user.id)


class TestListUserPortfolios:
    def test_returns_all_portfolios(self, session, user):
        create_portfolio(session, user.id, "B Portfolio", _POS)
        create_portfolio(session, user.id, "A Portfolio", _POS)
        session.commit()
        portfolios = list_user_portfolios(session, user.id)
        assert len(portfolios) == 2

    def test_ordered_by_name(self, session, user):
        create_portfolio(session, user.id, "Zebra", _POS)
        create_portfolio(session, user.id, "Alpha", _POS)
        session.commit()
        portfolios = list_user_portfolios(session, user.id)
        assert portfolios[0].name == "Alpha"
        assert portfolios[1].name == "Zebra"

    def test_isolated_from_other_user(self, session, user, other_user):
        create_portfolio(session, user.id, "Alice Portfolio", _POS)
        create_portfolio(session, other_user.id, "Bob Portfolio", _POS)
        session.commit()
        alice_portfolios = list_user_portfolios(session, user.id)
        assert len(alice_portfolios) == 1
        assert alice_portfolios[0].name == "Alice Portfolio"

    def test_empty_for_new_user(self, session, user):
        assert list_user_portfolios(session, user.id) == []


class TestUpdatePortfolioName:
    def test_renames(self, session, user):
        p = create_portfolio(session, user.id, "Old Name", _POS)
        session.commit()
        update_portfolio_name(session, p.id, user.id, "New Name")
        session.commit()
        loaded = session.get(Portfolio, p.id)
        assert loaded.name == "New Name"

    def test_empty_name_raises(self, session, user):
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        with pytest.raises(PortfolioValidationError):
            update_portfolio_name(session, p.id, user.id, "")

    def test_wrong_user_raises(self, session, user, other_user):
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        with pytest.raises(PortfolioNotFoundError):
            update_portfolio_name(session, p.id, other_user.id, "Hacked")


class TestSetPositions:
    def test_replaces_positions(self, session, user):
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        new_pos = [PositionInput(label="TLT", weight_pct=100.0, proxy_asset_id="us_treasuries")]
        set_positions(session, p.id, user.id, new_pos)
        session.commit()
        loaded = session.get(Portfolio, p.id)
        assert len(loaded.positions) == 1
        assert loaded.positions[0].label == "TLT"

    def test_old_positions_deleted(self, session, user):
        from sqlalchemy import select
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        old_ids = {pos.id for pos in p.positions}
        new_pos = [PositionInput(label="TLT", weight_pct=100.0, proxy_asset_id="us_treasuries")]
        set_positions(session, p.id, user.id, new_pos)
        session.commit()
        remaining = session.execute(
            select(PortfolioPosition).where(PortfolioPosition.id.in_(old_ids))
        ).scalars().all()
        assert remaining == []

    def test_empty_positions_raises(self, session, user):
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        with pytest.raises(PortfolioValidationError, match="at least"):
            set_positions(session, p.id, user.id, [])

    def test_wrong_user_raises(self, session, user, other_user):
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        with pytest.raises(PortfolioNotFoundError):
            set_positions(session, p.id, other_user.id, _POS)

    def test_updates_portfolio_updated_at(self, session, user):
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        # Strip tzinfo for comparison: SQLite returns naive datetimes on read.
        original = p.updated_at.replace(tzinfo=None)
        import time; time.sleep(0.01)
        set_positions(session, p.id, user.id, _POS)
        session.commit()
        loaded = session.get(Portfolio, p.id)
        loaded_dt = loaded.updated_at.replace(tzinfo=None) if loaded.updated_at.tzinfo else loaded.updated_at
        assert loaded_dt >= original


class TestDeletePortfolio:
    def test_deletes_portfolio(self, session, user):
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        delete_portfolio(session, p.id, user.id)
        session.commit()
        assert session.get(Portfolio, p.id) is None

    def test_cascade_deletes_positions(self, session, user):
        from sqlalchemy import select
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        pid = p.id
        delete_portfolio(session, pid, user.id)
        session.commit()
        remaining = session.execute(
            select(PortfolioPosition).where(PortfolioPosition.portfolio_id == pid)
        ).scalars().all()
        assert remaining == []

    def test_not_found_raises(self, session, user):
        with pytest.raises(PortfolioNotFoundError):
            delete_portfolio(session, "nonexistent", user.id)

    def test_wrong_user_raises(self, session, user, other_user):
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        with pytest.raises(PortfolioNotFoundError):
            delete_portfolio(session, p.id, other_user.id)


class TestCascadeDeleteUser:
    def test_deleting_user_removes_portfolios(self, session, user):
        create_portfolio(session, user.id, "P1", _POS)
        create_portfolio(session, user.id, "P2", _POS)
        session.commit()
        session.delete(user)
        session.commit()
        from sqlalchemy import select
        remaining = session.execute(
            select(Portfolio).where(Portfolio.user_id == user.id)
        ).scalars().all()
        assert remaining == []

    def test_deleting_user_removes_positions(self, session, user):
        from sqlalchemy import select
        p = create_portfolio(session, user.id, "P", _POS)
        session.commit()
        session.delete(user)
        session.commit()
        remaining = session.execute(
            select(PortfolioPosition).where(PortfolioPosition.portfolio_id == p.id)
        ).scalars().all()
        assert remaining == []
