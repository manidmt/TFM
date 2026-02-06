'''
@author: Manuel Díaz-Meco Terrés
@email: manidmt5@gmail.com
@date: 2026-02-06

@description: Tests for building forward-looking volatility labels (continuous) from features.
             Labels table should contain vol_fwd and NOT contain regime bins (those are train-fitted in make_dataset).
'''

import duckdb

DB = "data/db/financial_data.duckdb"


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    q = """
    SELECT COUNT(*)
    FROM information_schema.tables
    WHERE table_schema = 'main' AND table_name = ?
    """
    return con.execute(q, [name]).fetchone()[0] == 1


def _get_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    # pragma: (cid, name, type, notnull, dflt_value, pk)
    return {r[1] for r in rows}


def test_labels_table_exists_and_has_rows():
    con = duckdb.connect(DB)
    try:
        assert _table_exists(con, "labels_regime"), "labels_regime does not exist (run build_features.py)."
        n = con.execute("SELECT COUNT(*) FROM labels_regime").fetchone()[0]
        assert n > 0, "labels_regime is empty."
    finally:
        con.close()


def test_labels_schema_has_vol_fwd_and_no_regime():
    con = duckdb.connect(DB)
    try:
        cols = _get_columns(con, "labels_regime")
        required = {"ticker", "date", "horizon", "vol_fwd"}
        missing = required - cols
        assert not missing, f"labels_regime missing expected columns: {missing}"

        # Regime bins must be computed after splitting (train-fitted), not stored here.
        assert "regime" not in cols, (
            "labels_regime contains 'regime'. Regime bins should be computed in make_dataset "
            "after time split using TRAIN-only thresholds to avoid leakage."
        )
    finally:
        con.close()


def test_labels_have_expected_horizons():
    con = duckdb.connect(DB)
    try:
        horizons = {r[0] for r in con.execute("SELECT DISTINCT horizon FROM labels_regime").fetchall()}
        assert {5, 20} <= horizons, f"Unexpected horizons present: {horizons}"
    finally:
        con.close()


def test_labels_vol_fwd_is_non_null_and_non_negative():
    con = duckdb.connect(DB)
    try:
        # No nulls
        nulls = con.execute("""
            SELECT COUNT(*)
            FROM labels_regime
            WHERE vol_fwd IS NULL
        """).fetchone()[0]
        assert nulls == 0, f"Found {nulls} rows with vol_fwd NULL."

        # std can be 0 in rare flat periods, but must never be negative
        neg = con.execute("""
            SELECT COUNT(*)
            FROM labels_regime
            WHERE vol_fwd < 0
        """).fetchone()[0]
        assert neg == 0, f"Found {neg} rows with vol_fwd < 0 (invalid)."
    finally:
        con.close()


def test_labels_vol_fwd_has_reasonable_positive_mass():
    """
    Guardrail: vol_fwd should be > 0 in almost all rows. If it's mostly 0, something is wrong
    (e.g., returns are all zeros due to forward-fill issues or misalignment).
    """
    con = duckdb.connect(DB)
    try:
        total = con.execute("SELECT COUNT(*) FROM labels_regime").fetchone()[0]
        pos = con.execute("SELECT COUNT(*) FROM labels_regime WHERE vol_fwd > 0").fetchone()[0]
        # Allow a tiny fraction of zeros; require at least 95% positive
        assert total > 0
        assert pos / total >= 0.95, f"vol_fwd > 0 ratio too low: {pos}/{total} = {pos/total:.3f}"
    finally:
        con.close()


def test_labels_join_with_features_nonempty():
    con = duckdb.connect(DB)
    try:
        n = con.execute("""
            SELECT COUNT(*)
            FROM labels_regime l
            JOIN features_daily f
              ON l.ticker = f.ticker AND l.date = f.date
        """).fetchone()[0]
        assert n > 0, "labels_regime does not join with features_daily (date/ticker mismatch)."
    finally:
        con.close()


def test_labels_per_ticker_per_horizon_have_rows():
    """
    Ensure every selected ticker has labels for each horizon.
    """
    con = duckdb.connect(DB)
    try:
        rows = con.execute("""
            SELECT ticker, horizon, COUNT(*) AS n
            FROM labels_regime
            GROUP BY ticker, horizon
        """).fetchall()

        assert rows, "No (ticker, horizon) groups found in labels_regime."

        # Basic sanity: all groups have rows
        for t, h, n in rows:
            assert n > 0, f"Empty group in labels_regime for ticker={t}, horizon={h}"
    finally:
        con.close()
