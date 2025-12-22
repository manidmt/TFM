'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2025-12-22

@description: Module to fetch financial macroeconomic data from various sources.
'''

import pandas_datareader.data as web
import yfinance as yf
import pandas as pd
import duckdb
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MACRO_TICKERS = {
    'M2_USD': 'M2SL',
    'FED_ASSETS': 'WALCL',
    'US_10Y_YIELD': 'DGS10',
    'YIELD_CURVE': 'T10Y2Y'
}

DB_PATH = Path("data/db/macro_data.duckdb")

def init_db():
    """
    Docstring for init_db
    """
    con = duckdb.connect(database=str(DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS macro_features (
            date DATE PRIMARY KEY,
            M2_US DOUBLE,
            FED_ASSETS DOUBLE,
            US_10Y_YIELD DOUBLE,
            YIELD_CURVE DOUBLE,
            DXY_CLOSE DOUBLE
        )
    """)
    con.close()

def get_last_date():
    """
    Docstring for get_last_date
    """
    con = duckdb.connect(database=str(DB_PATH))
    try:
        result = con.execute("SELECT MAX(date) FROM macro_features").fetchone()
        last_date = result[0] if result else None
    except duckdb.CatalogException:
        last_date = None
    con.close()
    return last_date

def fetch_fred_data(start_date):
    """
    Docstring for fetch_fred_data
    """
    logging.info(f"Fetching FRED data on {start_date}")
    try:
        df = web.DataReader(list(MACRO_TICKERS.values()), 'fred', start_date, pd.Timestamp.now())
        df.rename(columns={v: k for k, v in MACRO_TICKERS.items()}, inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching FRED data: {e}")
        return pd.DataFrame()
    

def fetch_dxy_data(start_date):
    """
    Docstring for fetch_dxy_data
    """
    logging.info(f"Fetching DXY data on {start_date}")
    dxy = yf.download('DX-Y.NYB', start=start_date, auto_adjust=True, progress=False)
    if dxy.empty:
        logging.warning("No DXY data fetched.")
        return pd.DataFrame()
    if isinstance(dxy.index, pd.MultiIndex):
        dxy.columns = dxy.columns.get_level_values(0)
    
    df = pd.DataFrame()
    if 'Close' in dxy.columns:
        df['DXY_CLOSE'] = dxy['Close']
    else:
        df['DXY_CLOSE'] = dxy.iloc[:, 0]
        
    return df

def process_and_merge(df_fred, df_dxy):
    """
    Docstring for process_and_merge
    """
    if df_fred.empty and df_dxy.empty:
        logging.info("No data to process.")
        return pd.DataFrame()

    df_total = df_fred.join(df_dxy, how='outer')
    all_days = pd.date_range(start=df_total.index.min(), end=df_total.index.max(), freq='D')
    df_total = df_total.reindex(all_days)

    df_total = df_total.ffill()

    df_total = df_total.dropna()
    
    df_total.index.name = 'date'
    df_final = df_total.reset_index()
    df_final['date'] = pd.to_datetime(df_final['date']).dt.date
    return df_final

def store_data(df):
    """
    Docstring for store_data
    """
    if df.empty:
        logging.info("No data to store.")
        return

    con = duckdb.connect(database=str(DB_PATH))
    con.execute("CREATE TEMPORARY TABLE temp_macro AS SELECT * FROM macro_features WHERE 1=0")
    
    con.execute("INSERT INTO temp_macro SELECT * FROM df")

    con.execute("""
        INSERT INTO macro_features 
        SELECT * FROM temp_macro
        ON CONFLICT (date) DO UPDATE 
        SET M2_US = excluded.M2_US,
            FED_ASSETS = excluded.FED_ASSETS,
            US_10Y_YIELD = excluded.US_10Y_YIELD,
            YIELD_CURVE = excluded.YIELD_CURVE,
            DXY_CLOSE = excluded.DXY_CLOSE
    """)
    
    count = con.execute("SELECT COUNT(*) FROM macro_features").fetchone()[0]
    con.close()
    logging.info(f"Stored macro data. Total records: {count}")

def run():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db()
    
    last_date = get_last_date()
    
    if last_date:
        start_date = (last_date - timedelta(days=5)).strftime('%Y-%m-%d')
        logging.info(f"Incremental update from {start_date}")
    else:
        start_date = "2000-01-01"
        logging.info("Initial Full Load")

    df_fred = fetch_fred_data(start_date)
    df_dxy = fetch_dxy_data(start_date)
    
    if df_fred.empty and df_dxy.empty:
        logging.warning("No data found.")
        return

    df_final = process_and_merge(df_fred, df_dxy)
    
    store_data(df_final)

if __name__ == "__main__":
    run()