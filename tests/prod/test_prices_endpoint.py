'''Tests for GET /api/public/prices/history endpoint.'''
from __future__ import annotations

import duckdb
import pytest
from fastapi.testclient import TestClient

from quant_risk.prod.api.app import create_app
from quant_risk.prod.api.config import AppConfig
from quant_risk.prod.api.deps import get_auth_db, get_config, get_research_db, get_serving_db
from quant_risk.prod.auth.db import AuthDB
from quant_risk.prod.serving.duckdb import ServingDB


@pytest.fixture
def research_db():
    """In-memory DuckDB with raw_prices rows for bitcoin (BTC-USD) and gold (GLD)."""
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE raw_prices (
            ticker  VARCHAR NOT NULL,
            date    DATE    NOT NULL,
            open    DOUBLE,
            high    DOUBLE,
            low     DOUBLE,
            close   DOUBLE,
            volume  BIGINT,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        INSERT INTO raw_prices VALUES
        ('BTC-USD', '2025-10-01', 40000, 41000, 39000, 40500, 1000),
        ('BTC-USD', '2025-10-02', 40500, 42000, 40000, 41200, 1100),
        ('BTC-USD', '2025-10-03', 41200, 43000, 41000, 42800, 1200),
        ('GLD',     '2025-10-01', 180,   182,   179,   181,   5000),
        ('GLD',     '2025-10-02', 181,   183,   180,   182,   5100)
    """)
    yield conn
    conn.close()


@pytest.fixture
def client(research_db, tmp_path):
    """TestClient with research_db and minimal auth/serving stubs."""
    app = create_app()
    auth_db = AuthDB(f"sqlite:///{tmp_path}/auth.db")
    auth_db.create_tables()
    serving_db = ServingDB(":memory:")

    test_config = AppConfig()
    test_config.postgres_url = f"sqlite:///{tmp_path}/auth.db"
    test_config.serving_db_path = ":memory:"
    test_config.research_db_path = ":memory:"  # unused — overridden below

    def _auth_db_override():
        with auth_db.session() as s:
            try:
                yield s
            except Exception:
                s.rollback()
                raise
            finally:
                s.close()

    app.dependency_overrides[get_auth_db] = _auth_db_override
    app.dependency_overrides[get_serving_db] = lambda: (yield serving_db)
    app.dependency_overrides[get_research_db] = lambda: (yield research_db)
    app.dependency_overrides[get_config] = lambda: test_config

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    app.dependency_overrides.clear()
    serving_db.close()


def test_prices_history_returns_list(client):
    resp = client.get("/api/public/prices/history?asset_id=bitcoin&days=180")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 3


def test_prices_history_fields(client):
    resp = client.get("/api/public/prices/history?asset_id=bitcoin&days=180")
    row = resp.json()[0]
    assert set(row.keys()) == {"date", "close"}
    assert isinstance(row["date"], str)
    assert isinstance(row["close"], float)


def test_prices_history_ordered_asc(client):
    resp = client.get("/api/public/prices/history?asset_id=bitcoin&days=180")
    dates = [r["date"] for r in resp.json()]
    assert dates == sorted(dates)


def test_prices_history_unknown_asset_404(client):
    resp = client.get("/api/public/prices/history?asset_id=nonexistent&days=180")
    assert resp.status_code == 404


def test_prices_history_gold(client):
    resp = client.get("/api/public/prices/history?asset_id=gold&days=180")
    assert resp.status_code == 200
    assert len(resp.json()) == 2
