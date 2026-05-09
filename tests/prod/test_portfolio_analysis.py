'''
Tests for quant_risk.prod.portfolio_analysis

Uses a real in-memory ServingDB for prediction lookups and a real in-memory
AuthDB for ORM portfolios.  No mocks needed — both DBs are quick to set up.

Covers:
- analyze_portfolio: single asset, all predictions present
- analyze_portfolio: multiple assets, correct weighted signal
- analyze_portfolio: missing prediction → excluded from signal, in missing_predictions
- analyze_portfolio: all predictions missing → signal=None, probs all 0
- analyze_portfolio: weight normalization (unequal weights)
- analyze_portfolio: zero-weight positions handled (equal-share fallback)
- analyze_portfolio: empty positions raises ValueError
- portfolio_signal is "low" | "medium" | "high" per dominant prob
- asset_groups: one group per unique proxy_asset_id
- asset_groups: aggregate_weight sums correctly
- analyze_portfolio_orm: uses ORM Portfolio object
- analyze_portfolio_orm: what-if overrides positions
- _normalize_weights: normal / zero-total edge case
'''

import pytest
from datetime import date, datetime, timezone, timedelta
from unittest.mock import patch

import quant_risk.prod.auth.password as pw_mod
from quant_risk.prod.auth.db import AuthDB
from quant_risk.prod.auth.portfolios import PositionInput, create_portfolio
from quant_risk.prod.auth.users import create_user
from quant_risk.prod.portfolio_analysis import (
    PortfolioAnalysis,
    analyze_portfolio,
    analyze_portfolio_orm,
    _normalize_weights,
)
from quant_risk.prod.serving.duckdb import ServingDB
from quant_risk.prod.serving.predictions import persist_prediction
from quant_risk.prod.schemas import RunStatus

# Speed up scrypt
pw_mod.SCRYPT_N = 2

_FORECAST_DATE = date(2026, 3, 27)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_serving_db():
    db = ServingDB(":memory:")
    db.create_schema()
    return db


def _insert_prediction(
    db: ServingDB,
    asset_id: str,
    predicted_class: str,
    p_low: float,
    p_medium: float,
    p_high: float,
    forecast_date: date = _FORECAST_DATE,
):
    """Insert a minimal prediction row."""
    from quant_risk.prod.worker.inference import PredictionOutput
    from quant_risk.prod.schemas import PredictedClass

    class_map = {"low": PredictedClass.low, "medium": PredictedClass.medium, "high": PredictedClass.high}
    pred = PredictionOutput(
        predicted_class=class_map[predicted_class],
        p_low=p_low,
        p_medium=p_medium,
        p_high=p_high,
        bundle_version="2026-03-27_prod_abc",
    )
    persist_prediction(
        db=db,
        asset_id=asset_id,
        forecast_date=forecast_date,
        prediction=pred,
        status=RunStatus.fresh,
        data_cutoff_date=forecast_date,
    )


# ---------------------------------------------------------------------------
# _normalize_weights
# ---------------------------------------------------------------------------

class TestNormalizeWeights:
    def test_sums_to_one(self):
        positions = [
            PositionInput("A", 30.0, "us_equities"),
            PositionInput("B", 70.0, "btc"),
        ]
        nw = _normalize_weights(positions)
        assert abs(sum(nw) - 1.0) < 1e-9

    def test_proportional(self):
        positions = [
            PositionInput("A", 25.0, "us_equities"),
            PositionInput("B", 75.0, "btc"),
        ]
        nw = _normalize_weights(positions)
        assert abs(nw[0] - 0.25) < 1e-9
        assert abs(nw[1] - 0.75) < 1e-9

    def test_equal_weights(self):
        positions = [
            PositionInput("A", 50.0, "us_equities"),
            PositionInput("B", 50.0, "btc"),
        ]
        nw = _normalize_weights(positions)
        assert abs(nw[0] - 0.5) < 1e-9

    def test_zero_total_fallback_equal_share(self):
        positions = [
            PositionInput("A", 0.0, "us_equities"),
            PositionInput("B", 0.0, "btc"),
            PositionInput("C", 0.0, "us_treasuries"),
        ]
        nw = _normalize_weights(positions)
        assert abs(sum(nw) - 1.0) < 1e-9
        assert all(abs(w - 1 / 3) < 1e-9 for w in nw)


# ---------------------------------------------------------------------------
# analyze_portfolio
# ---------------------------------------------------------------------------

class TestAnalyzePortfolio:
    def test_single_asset_all_preds(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "high", 0.1, 0.2, 0.7)
        positions = [PositionInput("AAPL", 100.0, "us_equities")]
        result = analyze_portfolio("p1", "MyPort", positions, db)
        assert result.portfolio_signal == "high"
        assert abs(result.portfolio_p_high - 0.7) < 1e-6

    def test_two_assets_weighted_signal(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "low",    0.8, 0.1, 0.1)
        _insert_prediction(db, "btc",         "high",   0.1, 0.1, 0.8)
        # 80% equities (low signal) vs 20% btc (high signal)
        positions = [
            PositionInput("AAPL", 80.0, "us_equities"),
            PositionInput("BTC",  20.0, "btc"),
        ]
        result = analyze_portfolio("p1", "P", positions, db)
        # 0.8*0.8 + 0.2*0.1 = 0.64 + 0.02 = 0.66 -> dominant = low
        assert result.portfolio_signal == "low"
        assert abs(result.portfolio_p_low - 0.66) < 1e-5

    def test_missing_prediction_excluded_from_signal(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "medium", 0.2, 0.6, 0.2)
        # btc has no prediction
        positions = [
            PositionInput("AAPL", 50.0, "us_equities"),
            PositionInput("BTC",  50.0, "btc"),
        ]
        result = analyze_portfolio("p1", "P", positions, db)
        assert "btc" in result.missing_predictions
        # Signal computed from us_equities alone → medium
        assert result.portfolio_signal == "medium"

    def test_all_predictions_missing_signal_none(self):
        db = _make_serving_db()
        positions = [PositionInput("BTC", 100.0, "btc")]
        result = analyze_portfolio("p1", "P", positions, db)
        assert result.portfolio_signal is None
        assert result.portfolio_p_low == 0.0
        assert result.portfolio_p_medium == 0.0
        assert result.portfolio_p_high == 0.0
        assert "btc" in result.missing_predictions

    def test_empty_positions_raises(self):
        db = _make_serving_db()
        with pytest.raises(ValueError, match="no positions"):
            analyze_portfolio("p1", "P", [], db)

    def test_total_weight_pct_computed(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "low", 0.7, 0.2, 0.1)
        positions = [
            PositionInput("A", 30.0, "us_equities"),
            PositionInput("B", 40.0, "us_equities"),
        ]
        result = analyze_portfolio("p1", "P", positions, db)
        assert abs(result.total_weight_pct - 70.0) < 1e-9

    def test_per_position_normalized_weight(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "low", 0.7, 0.2, 0.1)
        positions = [
            PositionInput("A", 25.0, "us_equities"),
            PositionInput("B", 75.0, "us_equities"),
        ]
        result = analyze_portfolio("p1", "P", positions, db)
        nw = {pa.label: pa.normalized_weight for pa in result.positions}
        assert abs(nw["A"] - 0.25) < 1e-9
        assert abs(nw["B"] - 0.75) < 1e-9

    def test_asset_groups_one_per_proxy(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "low",  0.7, 0.2, 0.1)
        _insert_prediction(db, "btc",         "high", 0.1, 0.2, 0.7)
        positions = [
            PositionInput("AAPL", 40.0, "us_equities"),
            PositionInput("MSFT", 20.0, "us_equities"),
            PositionInput("BTC",  40.0, "btc"),
        ]
        result = analyze_portfolio("p1", "P", positions, db)
        assert len(result.asset_groups) == 2
        by_id = {g.asset_id: g for g in result.asset_groups}
        assert "us_equities" in by_id
        assert "btc" in by_id

    def test_asset_group_aggregate_weight(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "low", 0.7, 0.2, 0.1)
        positions = [
            PositionInput("AAPL", 40.0, "us_equities"),
            PositionInput("MSFT", 60.0, "us_equities"),
        ]
        result = analyze_portfolio("p1", "P", positions, db)
        assert len(result.asset_groups) == 1
        assert abs(result.asset_groups[0].aggregate_weight - 1.0) < 1e-9

    def test_asset_group_position_labels(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "low", 0.7, 0.2, 0.1)
        positions = [
            PositionInput("AAPL", 40.0, "us_equities"),
            PositionInput("MSFT", 60.0, "us_equities"),
        ]
        result = analyze_portfolio("p1", "P", positions, db)
        group = result.asset_groups[0]
        assert "AAPL" in group.position_labels
        assert "MSFT" in group.position_labels

    def test_position_analysis_has_prediction_fields(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "medium", 0.2, 0.6, 0.2)
        positions = [PositionInput("AAPL", 100.0, "us_equities")]
        result = analyze_portfolio("p1", "P", positions, db)
        pa = result.positions[0]
        assert pa.predicted_class == "medium"
        assert abs(pa.p_medium - 0.6) < 1e-6
        assert pa.forecast_date == _FORECAST_DATE

    def test_portfolio_signal_low(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "low", 0.8, 0.1, 0.1)
        positions = [PositionInput("AAPL", 100.0, "us_equities")]
        result = analyze_portfolio("p1", "P", positions, db)
        assert result.portfolio_signal == "low"

    def test_portfolio_signal_medium(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "medium", 0.1, 0.8, 0.1)
        positions = [PositionInput("AAPL", 100.0, "us_equities")]
        result = analyze_portfolio("p1", "P", positions, db)
        assert result.portfolio_signal == "medium"

    def test_portfolio_id_in_result(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "low", 0.7, 0.2, 0.1)
        positions = [PositionInput("AAPL", 100.0, "us_equities")]
        result = analyze_portfolio("my-portfolio-id", "P", positions, db)
        assert result.portfolio_id == "my-portfolio-id"

    def test_portfolio_name_in_result(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "low", 0.7, 0.2, 0.1)
        positions = [PositionInput("AAPL", 100.0, "us_equities")]
        result = analyze_portfolio("p1", "Alpha Portfolio", positions, db)
        assert result.portfolio_name == "Alpha Portfolio"

    def test_probs_sum_to_approx_one_with_all_predictions(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "low",  0.7, 0.2, 0.1)
        _insert_prediction(db, "btc",         "high", 0.1, 0.2, 0.7)
        positions = [
            PositionInput("AAPL", 50.0, "us_equities"),
            PositionInput("BTC",  50.0, "btc"),
        ]
        result = analyze_portfolio("p1", "P", positions, db)
        total = result.portfolio_p_low + result.portfolio_p_medium + result.portfolio_p_high
        assert abs(total - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# analyze_portfolio_orm
# ---------------------------------------------------------------------------

class TestAnalyzePortfolioOrm:
    @pytest.fixture
    def auth_db(self):
        db = AuthDB("sqlite:///:memory:")
        db.create_tables()
        yield db
        db.close()

    @pytest.fixture
    def serving_db(self):
        db = _make_serving_db()
        _insert_prediction(db, "us_equities", "low",  0.7, 0.2, 0.1)
        _insert_prediction(db, "btc",         "high", 0.1, 0.1, 0.8)
        yield db
        db.close()

    def test_uses_portfolio_positions(self, auth_db, serving_db):
        with auth_db.session() as s:
            u = create_user(s, "alice@example.com", "password1")
            s.commit()
            p = create_portfolio(s, u.id, "My Port",
                                 [PositionInput("AAPL", 100.0, "us_equities")])
            s.commit()
            result = analyze_portfolio_orm(p, serving_db)
        assert result.portfolio_id == p.id
        assert result.portfolio_signal == "low"

    def test_what_if_overrides_positions(self, auth_db, serving_db):
        with auth_db.session() as s:
            u = create_user(s, "alice@example.com", "password1")
            s.commit()
            p = create_portfolio(s, u.id, "My Port",
                                 [PositionInput("AAPL", 100.0, "us_equities")])
            s.commit()
            # What-if: swap to all btc
            overrides = [PositionInput("BTC", 100.0, "btc")]
            result = analyze_portfolio_orm(p, serving_db, position_overrides=overrides)
        assert result.portfolio_signal == "high"

    def test_portfolio_name_preserved(self, auth_db, serving_db):
        with auth_db.session() as s:
            u = create_user(s, "alice@example.com", "password1")
            s.commit()
            p = create_portfolio(s, u.id, "Named Portfolio",
                                 [PositionInput("AAPL", 100.0, "us_equities")])
            s.commit()
            result = analyze_portfolio_orm(p, serving_db)
        assert result.portfolio_name == "Named Portfolio"
