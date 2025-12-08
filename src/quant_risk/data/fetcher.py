'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2025-12-08

@description: Module to fetch financial data from various sources.
'''

import yfinance as yf
import pandas as pd
import duckdb
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ASSETS = {
    'Stocks': '^GSPC',  # S&P 500
    'Crypto': 'BTC-USD',  # Bitcoin
    'Bonds': 'TLT',  # iShares 20+ Year Treasury Bond ETF
    'Volatility': '^VIX'  # CBOE Volatility Index
}

DB_PATH = Path("data/db/financial_data.duckdb")

def init_db():
    """
    Docstring for init_db
    """
    con = duckdb.connect(database=str(DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_prices (
            ticker VARCHAR,
            date DATE,
            adj_close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (ticker, date)
        )
    """)
    con.close()

def get_last_date():
    """
    Docstring for get_last_date
    """
    con = duckdb.connect(database=str(DB_PATH))
    result = con.execute("SELECT MAX(date) FROM financial_data").fetchone()
    con.close()
    return result[0] if result[0] else None

def fetch_data():
    """
    Docstring for fetch_data
    """
    last_date = get_last_date()
    
    if last_date:
        start_date = (last_date - timedelta(days=2)).strftime('%Y-%m-%d')
        logging.info(f"Fetching data from {start_date} onwards.")
        period = None
    else:
        start_date = "2000-01-01"
        logging.info("No existing data found. Fetching all available data.")

    tickers = " ".join(ASSETS.values())

    if start_date:
        data = yf.download(tickers, start=start_date, group_by='ticker', auto_adjust=True, progress=False)
    else:
        data = yf.download(tickers, period='max', group_by='ticker', auto_adjust=True, progress=False)

    if data.empty:
        logging.info("No new data fetched.")
        return
    
    records = []

    for asset_name, ticker in ASSETS.items():
        if len(ASSETS) > 1:
            if ticker not in data.columns.levels[0]:
                logging.warning(f"No data found for {ticker}. Skipping.")
                continue
            df = data[ticker].copy()
        else:
            df = data.copy()

        df = df.reset_index()

        for _, row in df.iterrows():
            if pd.notna(row['Close']):
                records.append((ticker, row['Date'].date(), row['Close'], 
                                int(row['Volume']) if 'Volume' in row and pd.notna(row['Volume']) else None))
                
    return records

def store_data(records):
    """
    Docstring for store_data
    """
    if not records:
        logging.info("No records to store.")
        return

    con = duckdb.connect(database=str(DB_PATH))
    
    con.execute("CREATE TEMPORARY TABLE temp_prices AS SELECT * FROM raw_prices WHERE 1=0")
    con.executemany("INSERT INTO temp_prices VALUES (?, ?, ?, ?)", records)
    
    con.execute("""
        INSERT INTO raw_prices 
        SELECT * FROM temp_prices
        ON CONFLICT (date, ticker) DO UPDATE 
        SET adj_close = excluded.adj_close, volume = excluded.volume
    """)

    count = con.execute("SELECT COUNT(*) FROM raw_prices").fetchone()[0]
    con.close()
    logging.info(f"Stored {count} records into the database.")

if __name__ == "__main__":
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db()
    records = fetch_data()
    store_data(records)