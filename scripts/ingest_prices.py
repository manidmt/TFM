'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-09

@description: Script to ingest OHLCV prices into DuckDB.
'''

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from quant_risk.data.fetcher import PriceIngestConfig, refresh_prices


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest OHLCV prices into DuckDB.")
    parser.add_argument("--config", default="config/datasources.yaml")
    parser.add_argument("--db", default=None, help="Override DuckDB path")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers override")
    parser.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    db_path = args.db or cfg["db"]["path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    tickers = cfg["prices"]["tickers"]
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    ingest_cfg = PriceIngestConfig(
        db_path=db_path,
        default_start=cfg["prices"].get("default_start", "2010-01-01"),
        auto_adjust=bool(cfg["prices"].get("auto_adjust", True)),
        lookback_buffer_days=int(cfg["prices"].get("lookback_buffer_days", 5)),
    )

    res = refresh_prices(ingest_cfg, tickers=tickers, end=args.end)
    print(res)

    return 0 if not res.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())