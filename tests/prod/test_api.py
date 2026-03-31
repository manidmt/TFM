'''
Tests for the FastAPI application (quant_risk.prod.api).

Uses FastAPI's TestClient (synchronous) — no real Postgres or DuckDB
network connections.  Both AuthDB and ServingDB are injected as in-memory
instances via dependency overrides.

Covers:
────────
Health
  - GET /health → 200
  - GET /health/ready → 200 (both DBs reachable in-memory)

Public
  - GET /api/public/assets → 200 (empty list when config missing)
  - GET /api/public/predictions/latest → 200 list
  - GET /api/public/predictions/history → 200 list / 404 unknown asset

Auth
  - POST /api/auth/login → 200 + cookie on valid creds
  - POST /api/auth/login → 401 on bad creds
  - POST /api/auth/logout → 204 + cookie cleared
  - GET  /api/auth/me → 200 with user profile
  - GET  /api/auth/me → 401 when unauthenticated
  - POST /api/auth/change-password → 200 + new cookie
  - POST /api/auth/change-password → 400 wrong current password

Private (portfolios)
  - GET  /api/private/portfolios → 200 empty list
  - POST /api/private/portfolios → 201 with positions
  - GET  /api/private/portfolios/{id} → 200
  - GET  /api/private/portfolios/{id} → 404 unknown
  - PUT  /api/private/portfolios/{id} → 200 rename + positions
  - DELETE /api/private/portfolios/{id} → 204
  - POST /api/private/portfolios/{id}/analyze → 200 with signal

Admin
  - GET /api/admin/ops/summary → 200 with counts
  - GET /api/admin/ops/assets → 200 list
  - GET /api/admin/ops/promotions → 200 list
  - GET /api/admin/ops/summary → 403 for non-admin user
'''

import pytest
from datetime import date
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import quant_risk.prod.auth.password as pw_mod
from quant_risk.prod.api.app import create_app
from quant_risk.prod.api.config import AppConfig
from quant_risk.prod.api.deps import get_auth_db, get_config, get_serving_db
from quant_risk.prod.auth.db import AuthDB
from quant_risk.prod.auth.users import create_user
from quant_risk.prod.serving.duckdb import ServingDB
from quant_risk.prod.serving.predictions import persist_prediction
from quant_risk.prod.schemas import RunStatus

# Speed up scrypt for tests
pw_mod.SCRYPT_N = 2

_FORECAST_DATE = date(2026, 3, 27)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def auth_db_instance():
    db = AuthDB("sqlite:///:memory:")
    db.create_tables()
    yield db
    db.close()


@pytest.fixture(scope="module")
def serving_db_instance():
    sdb = ServingDB(":memory:")
    sdb.create_schema()
    yield sdb
    sdb.close()


@pytest.fixture(scope="module")
def test_config():
    cfg = AppConfig()
    cfg.cookie_secure = False  # allow non-HTTPS in tests
    cfg.postgres_url = "sqlite:///:memory:"
    cfg.serving_db_path = ":memory:"
    return cfg


@pytest.fixture(scope="module")
def client(auth_db_instance, serving_db_instance, test_config):
    """TestClient with overridden DB dependencies."""
    app = create_app(test_config)

    def _get_auth_db_override():
        with auth_db_instance.session() as s:
            yield s

    def _get_serving_db_override():
        yield serving_db_instance

    def _get_config_override():
        return test_config

    app.dependency_overrides[get_auth_db] = _get_auth_db_override
    app.dependency_overrides[get_serving_db] = _get_serving_db_override
    app.dependency_overrides[get_config] = _get_config_override

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="module")
def user_credentials(auth_db_instance):
    """Create a regular user; return (email, password)."""
    with auth_db_instance.session() as s:
        create_user(s, "testuser@example.com", "testpass1", must_change_password=False)
        s.commit()
    return "testuser@example.com", "testpass1"


@pytest.fixture(scope="module")
def admin_credentials(auth_db_instance):
    """Create an admin user; return (email, password)."""
    with auth_db_instance.session() as s:
        create_user(s, "admin@example.com", "adminpass1", role="admin", must_change_password=False)
        s.commit()
    return "admin@example.com", "adminpass1"


@pytest.fixture(scope="module")
def user_client(client, user_credentials):
    """TestClient with a logged-in regular user session cookie."""
    email, password = user_credentials
    resp = client.post("/api/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    # TestClient shares cookies automatically
    return client


@pytest.fixture(scope="module")
def admin_client(test_config, auth_db_instance, serving_db_instance, admin_credentials):
    """Separate TestClient for admin-only tests (avoids cookie sharing with user_client)."""
    app = create_app(test_config)

    def _get_auth_db_override():
        with auth_db_instance.session() as s:
            yield s

    def _get_serving_db_override():
        yield serving_db_instance

    def _get_config_override():
        return test_config

    app.dependency_overrides[get_auth_db] = _get_auth_db_override
    app.dependency_overrides[get_serving_db] = _get_serving_db_override
    app.dependency_overrides[get_config] = _get_config_override

    with TestClient(app, raise_server_exceptions=True) as c:
        email, password = admin_credentials
        resp = c.post("/api/auth/login", json={"email": email, "password": password})
        assert resp.status_code == 200, resp.text
        yield c


def _insert_prediction(sdb, asset_id, predicted_class, p_low, p_medium, p_high):
    from quant_risk.prod.worker.inference import PredictionOutput
    from quant_risk.prod.schemas import PredictedClass
    class_map = {"low": PredictedClass.low, "medium": PredictedClass.medium, "high": PredictedClass.high}
    pred = PredictionOutput(
        predicted_class=class_map[predicted_class],
        p_low=p_low, p_medium=p_medium, p_high=p_high,
        bundle_version="2026-03-27_prod_abc",
    )
    persist_prediction(
        db=sdb, asset_id=asset_id, forecast_date=_FORECAST_DATE,
        prediction=pred, status=RunStatus.fresh, data_cutoff_date=_FORECAST_DATE,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_liveness(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readiness(self, client):
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["checks"]["auth_db"] is True
        assert data["checks"]["serving_db"] is True


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

class TestPublic:
    def test_assets_returns_list(self, client):
        resp = client.get("/api/public/assets")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_predictions_latest_empty(self, client):
        resp = client.get("/api/public/predictions/latest")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_predictions_latest_with_data(self, client, serving_db_instance):
        _insert_prediction(serving_db_instance, "us_equities", "high", 0.1, 0.2, 0.7)
        resp = client.get("/api/public/predictions/latest")
        assert resp.status_code == 200
        by_id = {r["asset_id"]: r for r in resp.json()}
        assert "us_equities" in by_id
        assert by_id["us_equities"]["predicted_class"] == "high"

    def test_predictions_latest_no_bundle_version(self, client, serving_db_instance):
        _insert_prediction(serving_db_instance, "btc", "medium", 0.2, 0.6, 0.2)
        resp = client.get("/api/public/predictions/latest")
        for row in resp.json():
            assert "bundle_version" not in row
            assert "model_type" not in row

    def test_predictions_history_returns_list(self, client, serving_db_instance):
        resp = client.get("/api/public/predictions/history?asset_id=us_equities")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) >= 1

    def test_predictions_history_unknown_asset_404(self, client):
        resp = client.get("/api/public/predictions/history?asset_id=nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_login_valid_credentials(self, client, user_credentials):
        email, password = user_credentials
        resp = client.post("/api/auth/login", json={"email": email, "password": password})
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == email
        assert data["role"] == "user"

    def test_login_invalid_password(self, client, user_credentials):
        email, _ = user_credentials
        resp = client.post("/api/auth/login", json={"email": email, "password": "wrongpass"})
        assert resp.status_code == 401

    def test_login_unknown_email(self, client):
        resp = client.post("/api/auth/login", json={"email": "nobody@x.com", "password": "pass"})
        assert resp.status_code == 401

    def test_login_sets_cookie(self, client, user_credentials):
        email, password = user_credentials
        resp = client.post("/api/auth/login", json={"email": email, "password": password})
        assert "qr_session" in resp.cookies or "qr_session" in client.cookies

    def test_me_authenticated(self, user_client, user_credentials):
        email, _ = user_credentials
        resp = user_client.get("/api/auth/me")
        assert resp.status_code == 200
        assert resp.json()["email"] == email

    def test_me_unauthenticated(self, test_config, auth_db_instance, serving_db_instance):
        """Fresh client with no cookies → 401."""
        app = create_app(test_config)

        def _db():
            with auth_db_instance.session() as s:
                yield s

        app.dependency_overrides[get_auth_db] = _db
        app.dependency_overrides[get_serving_db] = lambda: (yield serving_db_instance)
        app.dependency_overrides[get_config] = lambda: test_config
        with TestClient(app) as c:
            resp = c.get("/api/auth/me")
        assert resp.status_code == 401

    def test_change_password_success(self, test_config, auth_db_instance, serving_db_instance):
        """Use a dedicated client to avoid polluting the shared user_client session."""
        with auth_db_instance.session() as s:
            create_user(s, "pwchange@example.com", "oldpasswd1", must_change_password=False)
            s.commit()

        app = create_app(test_config)

        def _db():
            with auth_db_instance.session() as s:
                yield s

        app.dependency_overrides[get_auth_db] = _db
        app.dependency_overrides[get_serving_db] = lambda: (yield serving_db_instance)
        app.dependency_overrides[get_config] = lambda: test_config

        with TestClient(app) as c:
            login_resp = c.post(
                "/api/auth/login",
                json={"email": "pwchange@example.com", "password": "oldpasswd1"},
            )
            assert login_resp.status_code == 200
            change_resp = c.post(
                "/api/auth/change-password",
                json={"current_password": "oldpasswd1", "new_password": "newpasswd1"},
            )
        assert change_resp.status_code == 200
        assert change_resp.json()["must_change_password"] is False

    def test_change_password_wrong_current(self, user_client):
        resp = user_client.post(
            "/api/auth/change-password",
            json={"current_password": "wrongcurrent", "new_password": "newpasswd1"},
        )
        assert resp.status_code == 400

    def test_logout(self, test_config, auth_db_instance, serving_db_instance, user_credentials):
        """Login → logout → /me should return 401."""
        email, password = user_credentials

        app = create_app(test_config)

        def _get_db():
            with auth_db_instance.session() as s:
                yield s

        app.dependency_overrides[get_auth_db] = _get_db
        app.dependency_overrides[get_serving_db] = lambda: (yield serving_db_instance)
        app.dependency_overrides[get_config] = lambda: test_config

        with TestClient(app) as c:
            c.post("/api/auth/login", json={"email": email, "password": password})
            logout_resp = c.post("/api/auth/logout")
            assert logout_resp.status_code == 204
            me_resp = c.get("/api/auth/me")
        assert me_resp.status_code == 401


# ---------------------------------------------------------------------------
# Private — portfolios
# ---------------------------------------------------------------------------

class TestPrivatePortfolios:
    def test_list_portfolios_empty(self, user_client):
        resp = user_client.get("/api/private/portfolios")
        assert resp.status_code == 200

    def test_create_portfolio(self, user_client):
        resp = user_client.post(
            "/api/private/portfolios",
            json={
                "name": "Test Portfolio",
                "positions": [
                    {"label": "AAPL", "weight_pct": 60.0, "proxy_asset_id": "us_equities"},
                    {"label": "BTC",  "weight_pct": 40.0, "proxy_asset_id": "btc"},
                ],
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Portfolio"
        assert len(data["positions"]) == 2

    def test_get_portfolio(self, user_client):
        # Create and then retrieve
        create_resp = user_client.post(
            "/api/private/portfolios",
            json={
                "name": "Get Test",
                "positions": [
                    {"label": "AAPL", "weight_pct": 100.0, "proxy_asset_id": "us_equities"},
                ],
            },
        )
        pid = create_resp.json()["portfolio_id"]
        resp = user_client.get(f"/api/private/portfolios/{pid}")
        assert resp.status_code == 200
        assert resp.json()["portfolio_id"] == pid

    def test_get_portfolio_not_found(self, user_client):
        resp = user_client.get("/api/private/portfolios/nonexistent-id")
        assert resp.status_code == 404

    def test_update_portfolio_name(self, user_client):
        create_resp = user_client.post(
            "/api/private/portfolios",
            json={
                "name": "Old Name",
                "positions": [{"label": "X", "weight_pct": 100.0, "proxy_asset_id": "btc"}],
            },
        )
        pid = create_resp.json()["portfolio_id"]
        resp = user_client.put(
            f"/api/private/portfolios/{pid}",
            json={"name": "New Name"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "New Name"

    def test_update_portfolio_positions(self, user_client):
        create_resp = user_client.post(
            "/api/private/portfolios",
            json={
                "name": "Update Pos",
                "positions": [{"label": "A", "weight_pct": 100.0, "proxy_asset_id": "btc"}],
            },
        )
        pid = create_resp.json()["portfolio_id"]
        resp = user_client.put(
            f"/api/private/portfolios/{pid}",
            json={
                "positions": [
                    {"label": "B", "weight_pct": 50.0, "proxy_asset_id": "us_equities"},
                    {"label": "C", "weight_pct": 50.0, "proxy_asset_id": "btc"},
                ]
            },
        )
        assert resp.status_code == 200
        labels = {p["label"] for p in resp.json()["positions"]}
        assert labels == {"B", "C"}

    def test_update_portfolio_empty_body_is_422(self, user_client):
        create_resp = user_client.post(
            "/api/private/portfolios",
            json={
                "name": "PUT Empty",
                "positions": [{"label": "X", "weight_pct": 100.0, "proxy_asset_id": "btc"}],
            },
        )
        pid = create_resp.json()["portfolio_id"]
        resp = user_client.put(f"/api/private/portfolios/{pid}", json={})
        assert resp.status_code == 422

    def test_delete_portfolio(self, user_client):
        create_resp = user_client.post(
            "/api/private/portfolios",
            json={
                "name": "To Delete",
                "positions": [{"label": "X", "weight_pct": 100.0, "proxy_asset_id": "btc"}],
            },
        )
        pid = create_resp.json()["portfolio_id"]
        del_resp = user_client.delete(f"/api/private/portfolios/{pid}")
        assert del_resp.status_code == 204
        get_resp = user_client.get(f"/api/private/portfolios/{pid}")
        assert get_resp.status_code == 404

    def test_analyze_portfolio(self, user_client, serving_db_instance):
        _insert_prediction(serving_db_instance, "us_equities", "low", 0.7, 0.2, 0.1)
        create_resp = user_client.post(
            "/api/private/portfolios",
            json={
                "name": "Analyze Test",
                "positions": [
                    {"label": "AAPL", "weight_pct": 100.0, "proxy_asset_id": "us_equities"},
                ],
            },
        )
        pid = create_resp.json()["portfolio_id"]
        resp = user_client.post(f"/api/private/portfolios/{pid}/analyze", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["portfolio_signal"] == "low"
        assert len(data["positions"]) == 1

    def test_analyze_portfolio_what_if(self, user_client, serving_db_instance):
        _insert_prediction(serving_db_instance, "btc", "high", 0.1, 0.1, 0.8)
        create_resp = user_client.post(
            "/api/private/portfolios",
            json={
                "name": "What-if Test",
                "positions": [
                    {"label": "AAPL", "weight_pct": 100.0, "proxy_asset_id": "us_equities"},
                ],
            },
        )
        pid = create_resp.json()["portfolio_id"]
        # What-if: override to btc
        resp = user_client.post(
            f"/api/private/portfolios/{pid}/analyze",
            json={
                "positions": [
                    {"label": "BTC", "weight_pct": 100.0, "proxy_asset_id": "btc"}
                ]
            },
        )
        assert resp.status_code == 200
        assert resp.json()["portfolio_signal"] == "high"

    def test_private_requires_auth(self, test_config, auth_db_instance, serving_db_instance):
        app = create_app(test_config)

        def _db():
            with auth_db_instance.session() as s:
                yield s

        app.dependency_overrides[get_auth_db] = _db
        app.dependency_overrides[get_serving_db] = lambda: (yield serving_db_instance)
        app.dependency_overrides[get_config] = lambda: test_config
        with TestClient(app) as c:
            resp = c.get("/api/private/portfolios")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

class TestAdmin:
    def test_ops_summary(self, admin_client):
        resp = admin_client.get("/api/admin/ops/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "asset_count" in data
        assert "fresh_count" in data

    def test_ops_assets(self, admin_client):
        resp = admin_client.get("/api/admin/ops/assets")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_ops_promotions(self, admin_client):
        resp = admin_client.get("/api/admin/ops/promotions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_ops_summary_forbidden_for_user(self, user_client):
        resp = user_client.get("/api/admin/ops/summary")
        assert resp.status_code == 403

    def test_ops_assets_forbidden_for_user(self, user_client):
        resp = user_client.get("/api/admin/ops/assets")
        assert resp.status_code == 403

    def test_ops_promotions_forbidden_for_user(self, user_client):
        resp = user_client.get("/api/admin/ops/promotions")
        assert resp.status_code == 403
