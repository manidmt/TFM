'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-10

@description: Tests for building features from prices and macro data.
'''

import duckdb

DB = "data/db/financial_data.duckdb"

def test_features_table_exists_and_has_rows():
    con = duckdb.connect(DB)
    try:
        n = con.execute("select count(*) from features_daily").fetchone()[0]
        assert n > 0
    finally:
        con.close()

def test_features_has_expected_tickers():
    con = duckdb.connect(DB)
    try:
        rows = con.execute("select distinct ticker from features_daily order by ticker").fetchall()
        tickers = [r[0] for r in rows]
        assert set(tickers) >= {"^GSPC", "BTC-USD", "TLT"}
    finally:
        con.close()

def test_features_calendar_is_business_day():
    con = duckdb.connect(DB)
    try:
        # weekend rows should be zero if you built on B-day calendar
        weekend = con.execute("""
            select count(*)
            from features_daily
            where strftime('%w', date) in ('0','6')
        """).fetchone()[0]
        assert weekend == 0
    finally:
        con.close()

def test_features_has_no_null_close_after_ffill():
    con = duckdb.connect(DB)
    try:
        n_null = con.execute("SELECT COUNT(*) FROM features_daily WHERE close IS NULL").fetchone()[0]
        assert n_null == 0, f"features_daily has {n_null} NULL close values."
    finally:
        con.close()


def test_features_has_required_columns():
    con = duckdb.connect(DB)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info('features_daily')").fetchall()]
        required = {"ticker", "date", "close", "logret", "rv_5", "rv_20"}
        missing = required - set(cols)
        assert not missing, f"features_daily missing columns: {missing}"
    finally:
        con.close()
