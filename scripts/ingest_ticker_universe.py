'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-04-11

@description: Bulk ingest prices for the ticker universe (~200 tickers).

Reads config/prod/ticker_universe.yaml and downloads 1.5 years of OHLCV
prices into DuckDB (raw_prices table) using yfinance. Tickers are processed
in batches to avoid rate-limiting.

Run on the workstation (not RPi5) since this is a one-time heavy download.

Usage
-----
    poetry run python scripts/ingest_ticker_universe.py
    poetry run python scripts/ingest_ticker_universe.py --dry-run
    poetry run python scripts/ingest_ticker_universe.py --batch-size 10 --start 2024-10-01
    poetry run python scripts/ingest_ticker_universe.py --skip-existing
'''

from __future__ import annotations

import argparse
import time
from pathlib import Path

from quant_risk.data.fetcher import (
    PriceIngestConfig,
    connect,
    download_ohlcv,
    get_last_dates,
    init_schema,
    upsert_prices,
)
from quant_risk.prod.ticker_universe import load_ticker_universe


_DEFAULT_DB = "data/db/financial_data.duckdb"
_DEFAULT_CATALOG = "config/prod/ticker_universe.yaml"
_DEFAULT_START = "2024-10-01"
_BATCH_SIZE = 20
_BATCH_PAUSE_SECS = 2.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest prices for the ticker universe into DuckDB.",
    )
    parser.add_argument(
        "--catalog", default=_DEFAULT_CATALOG,
        help="Path to ticker_universe.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--db", default=_DEFAULT_DB,
        help="DuckDB path (default: %(default)s)",
    )
    parser.add_argument(
        "--start", default=_DEFAULT_START,
        help="Start date YYYY-MM-DD (default: %(default)s)",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=_BATCH_SIZE,
        help="Tickers per yfinance batch download (default: %(default)s)",
    )
    parser.add_argument(
        "--pause", type=float, default=_BATCH_PAUSE_SECS,
        help="Seconds to wait between batches (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip tickers already present in raw_prices",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List tickers to ingest without downloading",
    )
    args = parser.parse_args()

    # Load catalog
    catalog = load_ticker_universe(args.catalog)
    all_tickers = [t.ticker for t in catalog]
    print(f"Ticker universe: {len(all_tickers)} tickers loaded from {args.catalog}")

    # Filter if --skip-existing
    if args.skip_existing:
        Path(args.db).parent.mkdir(parents=True, exist_ok=True)
        con = connect(args.db)
        init_schema(con, "raw_prices")
        existing = set(get_last_dates(con, "raw_prices").keys())
        con.close()
        tickers = [t for t in all_tickers if t not in existing]
        print(f"  Skipping {len(all_tickers) - len(tickers)} already in DB, {len(tickers)} to ingest")
    else:
        tickers = all_tickers

    if not tickers:
        print("Nothing to ingest.")
        return 0

    # Dry run: just list
    if args.dry_run:
        print(f"\n[DRY RUN] Would ingest {len(tickers)} tickers from {args.start}:")
        for i, t in enumerate(tickers, 1):
            print(f"  {i:3d}. {t}")
        return 0

    # Real ingestion
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    con = connect(args.db)
    init_schema(con, "raw_prices")

    total_inserted = 0
    errors: dict[str, str] = {}
    batches = [tickers[i:i + args.batch_size] for i in range(0, len(tickers), args.batch_size)]

    print(f"\nIngesting {len(tickers)} tickers in {len(batches)} batches of up to {args.batch_size}...")
    print(f"  Start: {args.start}  |  End: {args.end or 'today'}  |  DB: {args.db}\n")

    for batch_idx, batch in enumerate(batches, 1):
        print(f"Batch {batch_idx}/{len(batches)}: {', '.join(batch)}")
        try:
            df = download_ohlcv(
                batch,
                start=args.start,
                end=args.end,
                auto_adjust=True,
                timeout_seconds=60,
            )
            if df.empty:
                for t in batch:
                    errors[t] = "Empty download result"
                print(f"  -> empty result")
            else:
                n = upsert_prices(con, "raw_prices", df, source="yfinance")
                total_inserted += n
                downloaded_tickers = set(df["ticker"].unique())
                missing = set(batch) - downloaded_tickers
                for t in missing:
                    errors[t] = "Not returned by yfinance"
                print(f"  -> {n} rows inserted ({len(downloaded_tickers)} tickers)")
        except Exception as e:
            for t in batch:
                errors[t] = str(e)
            print(f"  -> ERROR: {e}")

        if batch_idx < len(batches):
            time.sleep(args.pause)

    con.close()

    print(f"\nDone. Total rows inserted: {total_inserted}")
    if errors:
        print(f"\nErrors ({len(errors)} tickers):")
        for t, msg in sorted(errors.items()):
            print(f"  {t}: {msg}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
