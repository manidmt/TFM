'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-08

@description: Script to ingest daily GDELT signals into DuckDB.
'''

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from quant_risk.data.gdelt import GdeltIngestConfig, refresh_gdelt


def load_yaml(path: str) -> dict:
    """
    Load a YAML file and return its parsed dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    """
    CLI entrypoint to ingest daily GDELT signals into DuckDB.
    """
    parser = argparse.ArgumentParser(description="Ingest daily GDELT signals into DuckDB.")
    parser.add_argument("--config", default="config/datasources.yaml")
    parser.add_argument("--db", default=None, help="Override DuckDB path")
    parser.add_argument("--start", default=None, help="Override ingest start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Optional ingest end date YYYY-MM-DD")
    parser.add_argument("--query", default=None, help="Override GDELT query expression")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    gdelt_cfg = cfg.get("gdelt", {})
    queries_cfg = gdelt_cfg.get("queries")
    queries = (
        {str(k): str(v) for k, v in queries_cfg.items()}
        if isinstance(queries_cfg, dict)
        else None
    )
    if args.query:
        queries = None

    db_path = args.db or cfg["db"]["path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    ingest_cfg = GdeltIngestConfig(
        db_path=db_path,
        table=str(gdelt_cfg.get("table", "gdelt_gkg_daily")),
        start=str(args.start or gdelt_cfg.get("start", "2018-01-01")),
        lookback_buffer_days=int(gdelt_cfg.get("lookback_buffer_days", 14)),
        publication_lag_bdays=int(gdelt_cfg.get("publication_lag_bdays", 1)),
        mode=str(gdelt_cfg.get("mode", "timeline")),
        keep_artlist_sample=bool(gdelt_cfg.get("keep_artlist_sample", False)),
        query=str(args.query or gdelt_cfg.get("query", "(finance OR market OR stocks OR bitcoin OR treasury)")),
        queries=queries,
        max_records_per_day=int(gdelt_cfg.get("max_records_per_day", 250)),
        timeout_seconds=int(gdelt_cfg.get("timeout_seconds", 30)),
    )

    if not bool(gdelt_cfg.get("enabled", True)):
        print({"status": "skipped", "reason": "gdelt.enabled is false"})
        return 0

    res = refresh_gdelt(ingest_cfg, end=args.end)
    print(res)
    return 0 if not res.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
