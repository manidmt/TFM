'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-11

@description: Script to ingest canonical GDELT GKG raw signals into DuckDB.
'''

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from quant_risk.data.gkg import GkgIngestConfig, refresh_gkg


def load_yaml(path: str) -> dict:
    """
    Load a YAML file and return its parsed dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    """
    CLI entrypoint to ingest GKG raw and rebuild daily news features.
    """
    parser = argparse.ArgumentParser(description="Ingest canonical GKG raw news into DuckDB.")
    parser.add_argument("--config", default="config/datasources.yaml")
    parser.add_argument("--db", default=None, help="Override DuckDB path")
    parser.add_argument("--start", default=None, help="Override start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Override end date YYYY-MM-DD")
    parser.add_argument("--topic_ids", default=None, help="Comma-separated topic ids")
    parser.add_argument("--timeout_seconds", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=None)
    parser.add_argument("--retry_backoff_seconds", type=float, default=None)
    parser.add_argument("--retry_max_sleep_seconds", type=float, default=None)
    parser.add_argument("--request_pause_seconds", type=float, default=None)
    parser.add_argument("--sample_interval_minutes", type=int, default=None)
    parser.add_argument("--max_files_per_day", type=int, default=None)
    parser.add_argument("--max_files_total", type=int, default=None)
    parser.add_argument("--store_raw", dest="store_raw", action="store_true")
    parser.add_argument("--no_store_raw", dest="store_raw", action="store_false")
    parser.set_defaults(store_raw=None)
    parser.add_argument("--force_refresh_existing", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    gkg_cfg = cfg.get("gkg", {})

    if not bool(gkg_cfg.get("enabled", False)):
        print({"status": "skipped", "reason": "gkg.enabled is false"})
        return 0

    db_path = args.db or cfg["db"]["path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    topics_cfg = gkg_cfg.get("topics", {})
    topics = (
        {str(k): str(v) for k, v in topics_cfg.items()}
        if isinstance(topics_cfg, dict)
        else None
    )

    if args.quiet:
        verbose = False
    elif args.verbose:
        verbose = True
    else:
        verbose = bool(gkg_cfg.get("verbose", True))

    ingest_cfg = GkgIngestConfig(
        db_path=db_path,
        raw_table=str(gkg_cfg.get("raw_table", "gdelt_gkg_raw")),
        daily_table=str(gkg_cfg.get("daily_table", "news_features_daily")),
        profile_name=str(gkg_cfg.get("profile_name", "gkg_v1_light")),
        profile_version=str(gkg_cfg.get("profile_version", "2026-03-13")),
        profile_mode=str(gkg_cfg.get("profile_mode", "light_daily")),
        start=str(args.start or gkg_cfg.get("start", "2018-01-01")),
        lookback_buffer_days=int(gkg_cfg.get("lookback_buffer_days", 14)),
        publication_lag_bdays=int(gkg_cfg.get("publication_lag_bdays", 1)),
        timeout_seconds=int(args.timeout_seconds or gkg_cfg.get("timeout_seconds", 30)),
        max_retries=int(args.max_retries or gkg_cfg.get("max_retries", 6)),
        retry_backoff_seconds=float(
            args.retry_backoff_seconds or gkg_cfg.get("retry_backoff_seconds", 2.0)
        ),
        retry_max_sleep_seconds=float(
            args.retry_max_sleep_seconds or gkg_cfg.get("retry_max_sleep_seconds", 120.0)
        ),
        request_pause_seconds=float(
            args.request_pause_seconds or gkg_cfg.get("request_pause_seconds", 0.25)
        ),
        sample_interval_minutes=int(
            args.sample_interval_minutes
            if args.sample_interval_minutes is not None
            else gkg_cfg.get("sample_interval_minutes", 0)
        ),
        max_files_per_day=int(
            args.max_files_per_day
            if args.max_files_per_day is not None
            else gkg_cfg.get("max_files_per_day", 0)
        ),
        max_files_total=int(
            args.max_files_total
            if args.max_files_total is not None
            else gkg_cfg.get("max_files_total", 0)
        ),
        store_raw=bool(
            args.store_raw
            if args.store_raw is not None
            else gkg_cfg.get("store_raw", True)
        ),
        masterfile_url=str(gkg_cfg.get("masterfile_url", "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt")),
        topics=topics,
        verbose=bool(verbose),
    )

    topic_ids = None
    if args.topic_ids:
        topic_ids = [s.strip() for s in str(args.topic_ids).split(",") if s.strip()]

    res = refresh_gkg(
        ingest_cfg,
        start=args.start,
        end=args.end,
        topic_ids=topic_ids,
        force_refresh_existing=bool(args.force_refresh_existing),
    )
    print(res)
    return 0 if not res.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
