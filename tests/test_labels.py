'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-10

@description: Tests for building labels from features.
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


def test_labels_table_exists_and_has_rows():
    con = duckdb.connect(DB)
    try:
        assert _table_exists(con, "labels_regime"), "labels_regime does not exist (run build_features.py)."
        n = con.execute("SELECT COUNT(*) FROM labels_regime").fetchone()[0]
        assert n > 0, "labels_regime is empty."
    finally:
        con.close()


def test_labels_regime_values_in_range():
    con = duckdb.connect(DB)
    try:
        bad = con.execute("""
            SELECT COUNT(*)
            FROM labels_regime
            WHERE regime IS NULL OR regime < 0 OR regime > 2
        """).fetchone()[0]
        assert bad == 0, f"Found {bad} invalid regime values."
    finally:
        con.close()


def test_labels_have_expected_horizons():
    con = duckdb.connect(DB)
    try:
        horizons = {r[0] for r in con.execute("SELECT DISTINCT horizon FROM labels_regime").fetchall()}
        assert {5, 20} <= horizons, f"Unexpected horizons: {horizons}"
    finally:
        con.close()


def test_labels_are_reasonably_balanced_per_horizon():
    con = duckdb.connect(DB)
    try:
        rows = con.execute("""
            SELECT horizon, regime, COUNT(*) AS n
            FROM labels_regime
            GROUP BY horizon, regime
            ORDER BY horizon, regime
        """).fetchall()

        by_h = {}
        for h, r, n in rows:
            by_h.setdefault(h, []).append(n)

        for h, counts in by_h.items():
            assert len(counts) >= 2, f"Horizon {h} has too few classes present: {counts}"
            mx, mn = max(counts), min(counts)
            assert mx <= 3 * mn, f"Horizon {h} extremely imbalanced: {counts}"
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
