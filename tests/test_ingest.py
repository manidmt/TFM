'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-10

@description: Tests for data ingestion into DuckDB.
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


def test_db_file_and_tables_exist():
    con = duckdb.connect(DB)
    try:
        assert _table_exists(con, "raw_prices"), "raw_prices does not exist (run ingest_prices.py)."
        assert _table_exists(con, "macro_features"), "macro_features does not exist (run ingest_macro.py)."
    finally:
        con.close()


def test_raw_prices_has_expected_tickers_and_nonzero_rows():
    con = duckdb.connect(DB)
    try:
        n = con.execute("SELECT COUNT(*) FROM raw_prices").fetchone()[0]
        assert n > 0, "raw_prices is empty."

        tickers = [r[0] for r in con.execute("SELECT DISTINCT ticker FROM raw_prices").fetchall()]
        expected = {"^GSPC", "TLT", "BTC-USD"}
        missing = expected - set(tickers)
        assert not missing, f"Missing tickers in raw_prices: {missing}"
    finally:
        con.close()


def test_raw_prices_date_ranges_are_sane():
    con = duckdb.connect(DB)
    try:
        rows = con.execute("""
            SELECT ticker, MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*) AS n
            FROM raw_prices
            GROUP BY ticker
            ORDER BY ticker
        """).fetchall()

        assert len(rows) >= 3

        for ticker, min_d, max_d, n in rows:
            assert n > 100, f"{ticker}: too few rows ({n})."
            assert min_d < max_d, f"{ticker}: min_date >= max_date."
    finally:
        con.close()


def test_macro_features_not_empty_and_has_core_columns():
    con = duckdb.connect(DB)
    try:
        n = con.execute("SELECT COUNT(*) FROM macro_features").fetchone()[0]
        assert n > 0, "macro_features is empty."

        cols = [r[1] for r in con.execute("PRAGMA table_info('macro_features')").fetchall()]
        # Ajusta si tu macro_features se llama distinto internamente
        required = {"date", "vix", "m2", "sofr", "ffr", "fed_assets", "tga", "rrp"}
        missing = required - set(cols)
        assert not missing, f"macro_features missing columns: {missing}"
    finally:
        con.close()


def test_macro_features_date_range_is_sane():
    con = duckdb.connect(DB)
    try:
        min_d, max_d = con.execute("SELECT MIN(date), MAX(date) FROM macro_features").fetchone()
        assert min_d is not None and max_d is not None
        assert min_d < max_d
    finally:
        con.close()