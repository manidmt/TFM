"""Rebuild labels_regime on the RPi5 from existing raw_prices.

Computes 5-day forward realized volatility (std of the next 5 daily returns)
for the 6 production proxy tickers and inserts any missing rows into labels_regime.
Run on the RPi5 when labels are stale — raw_prices must already be up to date.

Usage:
    python3 scripts/rebuild_labels_rpi5.py
"""

import sys
from datetime import date, timedelta

import duckdb
import numpy as np

DB_PATH = "/home/manidmt/Desktop/risk/db/financial_data.duckdb"
HORIZON = 5
PROXY_TICKERS = ["^GSPC", "^STOXX50E", "BTC-USD", "GLD", "TLT", "SHY"]

db = duckdb.connect(DB_PATH)

# Pull all closes for the proxy tickers (full history needed to get returns)
print("Loading raw_prices …")
rows = db.execute(
    """
    SELECT ticker, date, close
    FROM raw_prices
    WHERE ticker IN (SELECT unnest(?)) AND close IS NOT NULL
    ORDER BY ticker, date
    """,
    [PROXY_TICKERS],
).fetchall()

# Group by ticker
from collections import defaultdict
closes: dict[str, list[tuple]] = defaultdict(list)
for ticker, d, c in rows:
    closes[ticker].append((d, float(c)))

total_inserted = 0

for ticker in PROXY_TICKERS:
    data = closes.get(ticker, [])
    if len(data) < HORIZON + 2:
        print(f"  {ticker}: not enough raw_prices, skipping")
        continue

    dates = [r[0] for r in data]
    prices = np.array([r[1] for r in data])

    # Daily simple returns
    rets = prices[1:] / prices[:-1] - 1.0
    ret_dates = dates[1:]  # ret_dates[i] = return on ret_dates[i]

    # Forward 5-day std: for label date t, compute std(rets[t+1 .. t+HORIZON])
    # ret_dates index j corresponds to date ret_dates[j]
    # Forward window: indices j+1 .. j+HORIZON
    n = len(rets)
    label_rows = []
    for j in range(n - HORIZON):
        label_date = ret_dates[j]
        fwd_window = rets[j + 1 : j + 1 + HORIZON]
        vol_fwd = float(np.std(fwd_window, ddof=1))
        label_rows.append((ticker, label_date, HORIZON, vol_fwd))

    # Find last existing date in labels_regime for this ticker + horizon
    existing = db.execute(
        "SELECT MAX(date) FROM labels_regime WHERE ticker = ? AND horizon = ?",
        [ticker, HORIZON],
    ).fetchone()[0]

    # Only insert rows after the last existing date
    if existing is not None:
        new_rows = [(t, d, h, v) for t, d, h, v in label_rows if d > existing]
    else:
        new_rows = label_rows

    if not new_rows:
        print(f"  {ticker}: already up to date (last={existing})")
        continue

    # Exclude the last HORIZON rows (can't compute forward vol for them yet)
    # — already handled since the loop stops at n - HORIZON

    db.executemany(
        "INSERT INTO labels_regime (ticker, date, horizon, vol_fwd) VALUES (?, ?, ?, ?)",
        new_rows,
    )
    print(f"  {ticker}: inserted {len(new_rows)} rows "
          f"({new_rows[0][1]} → {new_rows[-1][1]})")
    total_inserted += len(new_rows)

db.commit()
db.close()
print(f"\nDone. Total rows inserted: {total_inserted}")
