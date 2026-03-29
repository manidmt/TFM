"""Tests for scripts/seed_dev_data.py — uses in-memory DuckDB."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import random

import pytest

# ---------------------------------------------------------------------------
# Import the script as a module (it lives in scripts/, not a package)
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).parents[2] / "scripts" / "seed_dev_data.py"
spec = importlib.util.spec_from_file_location("seed_dev_data", _SCRIPT)
_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(_mod)  # type: ignore[union-attr]

markov_next = _mod._markov_next
dirichlet_sample = _mod._dirichlet_sample
generate_rows = _mod._generate_rows
REGIMES = _mod.REGIMES
BASE_TRANSITIONS = _mod.BASE_TRANSITIONS
DIRICHLET_ALPHA = _mod.DIRICHLET_ALPHA
MARKER = _mod.MARKER
ASSETS = _mod.ASSETS


# ---------------------------------------------------------------------------
# _markov_next
# ---------------------------------------------------------------------------

def test_markov_next_returns_valid_regime():
    rng = random.Random(42)
    result = markov_next("low", BASE_TRANSITIONS, rng)
    assert result in REGIMES


def test_markov_next_deterministic_with_same_seed():
    rng1 = random.Random(99)
    rng2 = random.Random(99)
    r1 = markov_next("medium", BASE_TRANSITIONS, rng1)
    r2 = markov_next("medium", BASE_TRANSITIONS, rng2)
    assert r1 == r2


def test_markov_next_zero_prob_transition_never_happens():
    # low→high has prob 0.0 in BASE_TRANSITIONS
    rng = random.Random(0)
    results = {markov_next("low", BASE_TRANSITIONS, rng) for _ in range(200)}
    assert "high" not in results


# ---------------------------------------------------------------------------
# _dirichlet_sample
# ---------------------------------------------------------------------------

def test_dirichlet_sample_sums_to_one():
    rng = random.Random(7)
    p_low, p_med, p_high = dirichlet_sample(DIRICHLET_ALPHA["low"], rng)
    assert abs(p_low + p_med + p_high - 1.0) < 1e-9


def test_dirichlet_sample_all_positive():
    rng = random.Random(7)
    for regime in REGIMES:
        p = dirichlet_sample(DIRICHLET_ALPHA[regime], rng)
        assert all(v > 0 for v in p)


def test_dirichlet_low_regime_dominant():
    rng = random.Random(42)
    # average over many samples — p_low should dominate
    total_low = sum(dirichlet_sample(DIRICHLET_ALPHA["low"], rng)[0] for _ in range(500))
    assert total_low / 500 > 0.5


def test_dirichlet_high_regime_dominant():
    rng = random.Random(42)
    total_high = sum(dirichlet_sample(DIRICHLET_ALPHA["high"], rng)[2] for _ in range(500))
    assert total_high / 500 > 0.5


# ---------------------------------------------------------------------------
# _generate_rows
# ---------------------------------------------------------------------------

def test_generate_rows_count():
    rows = generate_rows("us_equities", 30)
    assert len(rows) == 31   # 30 days back + today


def test_generate_rows_fields():
    rows = generate_rows("us_equities", 10)
    required = {"asset_id", "forecast_date", "predicted_class",
                "p_low", "p_medium", "p_high", "bundle_version",
                "data_cutoff_date", "status", "updated_at"}
    assert required.issubset(rows[0].keys())


def test_generate_rows_marker():
    rows = generate_rows("bitcoin", 5)
    assert all(r["bundle_version"] == MARKER for r in rows)


def test_generate_rows_probs_sum_to_one():
    rows = generate_rows("gold", 20)
    for r in rows:
        assert abs(r["p_low"] + r["p_medium"] + r["p_high"] - 1.0) < 1e-9


def test_generate_rows_reproducible():
    rows1 = generate_rows("us_equities", 10)
    rows2 = generate_rows("us_equities", 10)
    assert [r["predicted_class"] for r in rows1] == [r["predicted_class"] for r in rows2]


def test_generate_rows_dates_ascending():
    rows = generate_rows("gold", 10)
    dates = [r["forecast_date"] for r in rows]
    assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# seed / wipe (in-memory DuckDB)
# ---------------------------------------------------------------------------

seed_fn = _mod.seed
wipe_fn = _mod.wipe
_DDL = _mod._DDL
import tempfile, os

def _make_tmp_db() -> str:
    """Return path to a fresh temp DuckDB file."""
    fd, path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    os.unlink(path)  # DuckDB needs a non-existent path to create a new DB
    return path


def test_seed_inserts_rows():
    import duckdb
    path = _make_tmp_db()
    try:
        n = seed_fn(path, n_days=10)
        assert n == 11 * len(ASSETS)   # 11 rows per asset (10 days + today)
        con = duckdb.connect(path)
        count = con.execute("SELECT COUNT(*) FROM predictions_daily").fetchone()[0]
        con.close()
        assert count == n
    finally:
        os.unlink(path)


def test_seed_all_rows_marked():
    import duckdb
    path = _make_tmp_db()
    try:
        seed_fn(path, n_days=5)
        con = duckdb.connect(path)
        unmarked = con.execute(
            f"SELECT COUNT(*) FROM predictions_daily WHERE bundle_version != '{MARKER}'"
        ).fetchone()[0]
        con.close()
        assert unmarked == 0
    finally:
        os.unlink(path)


def test_seed_idempotent():
    import duckdb
    path = _make_tmp_db()
    try:
        n1 = seed_fn(path, n_days=5)
        n2 = seed_fn(path, n_days=5)
        con = duckdb.connect(path)
        count = con.execute("SELECT COUNT(*) FROM predictions_daily").fetchone()[0]
        con.close()
        assert count == n1 == n2
    finally:
        os.unlink(path)


def test_wipe_removes_synthetic_rows():
    import duckdb
    path = _make_tmp_db()
    try:
        seed_fn(path, n_days=5)
        wiped = wipe_fn(path)
        assert wiped == 6 * 6   # 6 assets × 6 rows each
        con = duckdb.connect(path)
        count = con.execute("SELECT COUNT(*) FROM predictions_daily").fetchone()[0]
        con.close()
        assert count == 0
    finally:
        os.unlink(path)


def test_wipe_on_empty_db_returns_zero():
    path = _make_tmp_db()
    try:
        import duckdb
        con = duckdb.connect(path)
        con.execute(_DDL)
        con.close()
        wiped = wipe_fn(path)
        assert wiped == 0
    finally:
        os.unlink(path)
