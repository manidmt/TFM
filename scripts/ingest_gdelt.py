'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-08

@description: Script to ingest legacy GDELT DOC timeline signals into DuckDB.
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
    CLI entrypoint to ingest legacy daily GDELT DOC timeline signals into DuckDB.
    """
    parser = argparse.ArgumentParser(description="Ingest daily GDELT signals into DuckDB.")
    parser.add_argument("--config", default="config/datasources.yaml")
    parser.add_argument("--db", default=None, help="Override DuckDB path")
    parser.add_argument("--start", default=None, help="Override ingest start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Optional ingest end date YYYY-MM-DD")
    parser.add_argument("--query", default=None, help="Override GDELT query expression")
    parser.add_argument("--query_ids", default=None, help="Comma-separated query_ids to ingest")
    parser.add_argument("--mode", default=None, choices=["timeline", "artlist"], help="Override gdelt mode")
    parser.add_argument("--timeline_chunk_days", type=int, default=None, help="Chunk size in days for timeline mode.")
    parser.add_argument("--request_pause_seconds", type=float, default=None, help="Pause between API calls (seconds)")
    parser.add_argument("--max_retries", type=int, default=None, help="Max retries per HTTP request")
    parser.add_argument("--retry_backoff_seconds", type=float, default=None, help="Base exponential backoff (seconds)")
    parser.add_argument("--retry_max_sleep_seconds", type=float, default=None, help="Cap for retry sleep (seconds)")
    parser.add_argument("--timeout_seconds", type=int, default=None, help="HTTP timeout (seconds)")
    parser.add_argument("--verbose", action="store_true", help="Enable progress logging.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    gdelt_cfg = cfg.get("gdelt", {})
    queries_cfg = gdelt_cfg.get("queries")
    queries = (
        {str(k): str(v) for k, v in queries_cfg.items()}
        if isinstance(queries_cfg, dict)
        else None
    )
    query_candidates_cfg = gdelt_cfg.get("query_candidates")
    query_candidates = None
    if isinstance(query_candidates_cfg, dict):
        qc_out: dict[str, list[str]] = {}
        for qid, variants in query_candidates_cfg.items():
            if isinstance(variants, (list, tuple)):
                vals = [str(v).strip() for v in variants if str(v).strip()]
            else:
                vals = [str(variants).strip()] if str(variants).strip() else []
            if vals:
                qc_out[str(qid).strip()] = vals
        query_candidates = qc_out or None
    if args.query:
        queries = None
        query_candidates = None

    db_path = args.db or cfg["db"]["path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    if args.quiet:
        verbose = False
    elif args.verbose:
        verbose = True
    else:
        verbose = bool(gdelt_cfg.get("verbose", True))

    ingest_cfg = GdeltIngestConfig(
        db_path=db_path,
        table=str(gdelt_cfg.get("table", "gdelt_gkg_daily")),
        start=str(args.start or gdelt_cfg.get("start", "2018-01-01")),
        lookback_buffer_days=int(gdelt_cfg.get("lookback_buffer_days", 14)),
        publication_lag_bdays=int(gdelt_cfg.get("publication_lag_bdays", 1)),
        mode=str(args.mode or gdelt_cfg.get("mode", "timeline")),
        keep_artlist_sample=bool(gdelt_cfg.get("keep_artlist_sample", False)),
        query=str(args.query or gdelt_cfg.get("query", "(finance OR market OR stocks OR bitcoin OR treasury)")),
        queries=queries,
        query_candidates=query_candidates,
        query_dead_min_nonzero_rate=float(gdelt_cfg.get("query_dead_min_nonzero_rate", 0.01)),
        query_dead_min_avg_news_count=float(gdelt_cfg.get("query_dead_min_avg_news_count", 1.0)),
        max_records_per_day=int(gdelt_cfg.get("max_records_per_day", 250)),
        timeout_seconds=int(args.timeout_seconds or gdelt_cfg.get("timeout_seconds", 30)),
        timeline_chunk_days=int(args.timeline_chunk_days or gdelt_cfg.get("timeline_chunk_days", 30)),
        max_retries=int(args.max_retries or gdelt_cfg.get("max_retries", 6)),
        retry_backoff_seconds=float(
            args.retry_backoff_seconds or gdelt_cfg.get("retry_backoff_seconds", 2.0)
        ),
        retry_max_sleep_seconds=float(
            args.retry_max_sleep_seconds or gdelt_cfg.get("retry_max_sleep_seconds", 120.0)
        ),
        request_pause_seconds=float(
            args.request_pause_seconds or gdelt_cfg.get("request_pause_seconds", 0.25)
        ),
        verbose=bool(verbose),
    )

    print(
        "[deprecated] ingest_gdelt.py uses legacy DOC timeline ingestion. "
        "Use scripts/ingest_gkg.py for the canonical pipeline."
    )

    if not bool(gdelt_cfg.get("enabled", True)):
        print({"status": "skipped", "reason": "gdelt.enabled is false"})
        return 0

    query_ids = None
    if args.query_ids:
        query_ids = [s.strip() for s in str(args.query_ids).split(",") if s.strip()]

    res = refresh_gdelt(
        ingest_cfg,
        start=args.start,
        end=args.end,
        query_ids=query_ids,
    )
    print(res)
    return 0 if not res.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
