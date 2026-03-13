'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-09

@description: Missing-only GDELT backfill runner that requests only missing query/date ranges.
'''

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from quant_risk.data.gdelt import (
    GdeltIngestConfig,
    connect,
    fetch_artlist_range_for_query,
    fetch_timeline_range_for_query,
    init_schema,
    upsert_gdelt_daily,
)


def _log(msg: str) -> None:
    """
    Print a timestamped progress line.
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[gdelt-backfill] {ts} {msg}", flush=True)


def load_yaml(path: str) -> dict:
    """
    Load a YAML file into a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_queries(gdelt_cfg: dict, cli_query: str | None) -> dict[str, str]:
    """
    Resolve thematic queries to a normalized {query_id: query_text} mapping.
    """
    if cli_query:
        return {"default": str(cli_query)}

    queries_cfg = gdelt_cfg.get("queries")
    if isinstance(queries_cfg, dict):
        out: dict[str, str] = {}
        for k, v in queries_cfg.items():
            qid = str(k).strip()
            qtxt = str(v).strip()
            if not qid or not qtxt:
                continue
            out[qid] = qtxt
        if out:
            return out

    return {"default": str(gdelt_cfg.get("query", "(finance OR market OR stocks OR bitcoin OR treasury)"))}


def _iter_days(start_day: date, end_day: date) -> list[date]:
    """
    Build an inclusive list of calendar days.
    """
    if end_day < start_day:
        return []
    n_days = (end_day - start_day).days
    return [start_day + timedelta(days=i) for i in range(n_days + 1)]


def _get_existing_dates(
    con: duckdb.DuckDBPyConnection,
    table: str,
    query_id: str,
    start_day: date,
    end_day: date,
) -> set[date]:
    """
    Fetch existing dates for one query_id in [start_day, end_day].
    """
    rows = con.execute(
        f"""
        SELECT DISTINCT date::DATE
        FROM {table}
        WHERE query_id = ?
          AND date BETWEEN ?::DATE AND ?::DATE
        """,
        [str(query_id), start_day.isoformat(), end_day.isoformat()],
    ).fetchall()
    return {r[0] for r in rows}


def _missing_days(
    con: duckdb.DuckDBPyConnection,
    table: str,
    query_id: str,
    start_day: date,
    end_day: date,
) -> list[date]:
    """
    Return sorted missing days for one query_id in [start_day, end_day].
    """
    existing = _get_existing_dates(con, table, query_id, start_day, end_day)
    return [d for d in _iter_days(start_day, end_day) if d not in existing]


def _split_missing_ranges(missing: list[date], max_range_days: int) -> list[tuple[date, date]]:
    """
    Split missing days into contiguous ranges capped by max_range_days.
    """
    if not missing:
        return []

    out: list[tuple[date, date]] = []
    cap = max(1, int(max_range_days))
    start = missing[0]
    prev = missing[0]

    for d in missing[1:]:
        contiguous = d == (prev + timedelta(days=1))
        curr_len = (prev - start).days + 1
        if contiguous and curr_len < cap:
            prev = d
            continue
        out.append((start, prev))
        start = d
        prev = d

    out.append((start, prev))
    return out


def _fetch_query_range(
    cfg: GdeltIngestConfig,
    query_text: str,
    start_day: date,
    end_day: date,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Fetch one range for one query_id according to cfg.mode.
    """
    mode = str(cfg.mode).strip().lower()
    if mode == "timeline":
        return fetch_timeline_range_for_query(cfg, start_day, end_day, query_text=query_text)
    if mode == "artlist":
        return fetch_artlist_range_for_query(cfg, start_day, end_day, query_text=query_text)
    raise ValueError(f"Unsupported gdelt.mode={cfg.mode!r}. Use 'timeline' or 'artlist'.")


def _densify_daily(df: pd.DataFrame, start_day: date, end_day: date) -> pd.DataFrame:
    """
    Densify a daily dataframe to all calendar days in [start_day, end_day].
    """
    cal = pd.DataFrame({"date": _iter_days(start_day, end_day)})
    cols = ["date", "news_count", "tone_mean", "tone_std", "tone_neg_share"]

    if df is None or df.empty:
        out = cal.copy()
    else:
        keep = [c for c in cols if c in df.columns]
        out = cal.merge(df[keep].copy(), on="date", how="left")

    for c in ["news_count", "tone_mean", "tone_std", "tone_neg_share"]:
        if c not in out.columns:
            out[c] = pd.NA
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["news_count"] = out["news_count"].fillna(0.0)
    return out[cols]


def _backfill_query_once(
    con: duckdb.DuckDBPyConnection,
    cfg: GdeltIngestConfig,
    query_id: str,
    query_text: str,
    start_day: date,
    end_day: date,
    max_range_days: int,
    dry_run: bool,
) -> dict:
    """
    Backfill one pass for a query_id using only currently-missing ranges.
    """
    missing_before = _missing_days(con, cfg.table, query_id, start_day, end_day)
    ranges = _split_missing_ranges(missing_before, max_range_days=max_range_days)
    missing_set = set(missing_before)
    inserted_rows = 0
    errors: dict[str, str] = {}

    _log(
        f"query={query_id} missing_before={len(missing_before)} ranges={len(ranges)} "
        f"range={start_day}..{end_day}"
    )

    for idx, (r_start, r_end) in enumerate(ranges, start=1):
        _log(f"query={query_id} range_start {idx}/{len(ranges)} {r_start}..{r_end}")
        try:
            fetched, fetch_errors = _fetch_query_range(cfg, query_text=query_text, start_day=r_start, end_day=r_end)
            for k, msg in fetch_errors.items():
                errors[f"{query_id}:{k}"] = str(msg)

            if bool(fetch_errors):
                # On partial/failed fetches keep only observed rows and leave unresolved days for later rounds.
                frame = fetched.copy() if fetched is not None else pd.DataFrame()
            else:
                # On clean fetches densify to full range to store explicit zero-news days.
                frame = _densify_daily(fetched, r_start, r_end)

            if frame is None or frame.empty:
                _log(f"query={query_id} range_done {idx}/{len(ranges)} rows=0 errors={len(fetch_errors)}")
                continue

            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
            frame = frame[frame["date"].isin(missing_set)].dropna(subset=["date"])
            if frame.empty:
                _log(f"query={query_id} range_done {idx}/{len(ranges)} rows=0 errors={len(fetch_errors)}")
                continue

            frame["query_id"] = str(query_id)

            if dry_run:
                inserted_rows += int(frame.shape[0])
            else:
                inserted_rows += int(
                    upsert_gdelt_daily(
                        con,
                        cfg.table,
                        frame,
                        source="gdelt_doc_api_missing_backfill",
                    )
                )
            _log(
                f"query={query_id} range_done {idx}/{len(ranges)} rows={int(frame.shape[0])} "
                f"errors={len(fetch_errors)}"
            )
        except Exception as exc:
            errors[f"{query_id}:{r_start.isoformat()}..{r_end.isoformat()}"] = str(exc)
            _log(f"query={query_id} range_error {idx}/{len(ranges)} {r_start}..{r_end}: {exc}")

    missing_after = _missing_days(con, cfg.table, query_id, start_day, end_day)
    return {
        "query_id": query_id,
        "missing_before": len(missing_before),
        "missing_after": len(missing_after),
        "ranges": len(ranges),
        "inserted_rows": int(inserted_rows),
        "errors": errors,
    }


def main() -> int:
    """
    CLI entrypoint for missing-only GDELT backfill.
    """
    parser = argparse.ArgumentParser(description="Backfill GDELT only for missing query/date rows.")
    parser.add_argument("--config", default="config/datasources.yaml")
    parser.add_argument("--db", default=None, help="Override DuckDB path")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--query", default=None, help="Single query override")
    parser.add_argument("--query_ids", default=None, help="Comma-separated query_ids to backfill")
    parser.add_argument("--timeline_chunk_days", type=int, default=None, help="Chunk size for timeline mode requests")
    parser.add_argument("--request_pause_seconds", type=float, default=None, help="Pause between API calls (seconds)")
    parser.add_argument("--max_retries", type=int, default=None, help="Max retries per HTTP request")
    parser.add_argument("--retry_backoff_seconds", type=float, default=None, help="Base exponential backoff (seconds)")
    parser.add_argument("--retry_max_sleep_seconds", type=float, default=None, help="Cap for retry sleep (seconds)")
    parser.add_argument("--timeout_seconds", type=int, default=None, help="HTTP timeout (seconds)")
    parser.add_argument("--max_range_days", type=int, default=30, help="Max missing-range size per request")
    parser.add_argument("--max_rounds", type=int, default=6, help="Retry rounds over unresolved missing days")
    parser.add_argument("--sleep_between_rounds", type=float, default=60.0, help="Sleep seconds between rounds")
    parser.add_argument("--dry_run", action="store_true", help="Compute missing ranges only, do not write DB")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose ingestion logs")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose ingestion logs")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    gdelt_cfg = cfg.get("gdelt", {})

    if not bool(gdelt_cfg.get("enabled", True)):
        print({"status": "skipped", "reason": "gdelt.enabled is false"})
        return 0

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
        mode=str(gdelt_cfg.get("mode", "timeline")),
        keep_artlist_sample=bool(gdelt_cfg.get("keep_artlist_sample", False)),
        query=str(args.query or gdelt_cfg.get("query", "(finance OR market OR stocks OR bitcoin OR treasury)")),
        queries=_resolve_queries(gdelt_cfg, cli_query=args.query),
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

    start_day = pd.to_datetime(args.start or ingest_cfg.start).date()
    end_day = pd.to_datetime(args.end).date() if args.end else date.today()
    if end_day < start_day:
        raise ValueError(f"Invalid range: start={start_day} end={end_day}")

    queries = dict(ingest_cfg.queries or {"default": ingest_cfg.query})
    if args.query_ids:
        keep = {s.strip() for s in str(args.query_ids).split(",") if s.strip()}
        queries = {k: v for k, v in queries.items() if k in keep}
        if not queries:
            raise ValueError(f"--query_ids={args.query_ids!r} does not match configured query ids.")

    con = connect(ingest_cfg.db_path)
    try:
        init_schema(con, ingest_cfg.table)
        _log(
            f"start mode={ingest_cfg.mode} range={start_day}..{end_day} queries={sorted(queries.keys())} "
            f"max_rounds={int(args.max_rounds)} max_range_days={int(args.max_range_days)} dry_run={bool(args.dry_run)}"
        )

        all_errors: dict[str, str] = {}
        total_inserted = 0

        for round_idx in range(1, int(args.max_rounds) + 1):
            round_inserted = 0
            round_errors: dict[str, str] = {}
            per_query: list[dict] = []

            total_missing_before = 0
            for qid in queries:
                total_missing_before += len(_missing_days(con, ingest_cfg.table, qid, start_day, end_day))

            _log(f"round={round_idx} missing_before_total={total_missing_before}")
            if total_missing_before == 0:
                break

            for qid, qtxt in queries.items():
                stats = _backfill_query_once(
                    con=con,
                    cfg=ingest_cfg,
                    query_id=str(qid),
                    query_text=str(qtxt),
                    start_day=start_day,
                    end_day=end_day,
                    max_range_days=int(args.max_range_days),
                    dry_run=bool(args.dry_run),
                )
                per_query.append(stats)
                round_inserted += int(stats["inserted_rows"])
                round_errors.update(stats["errors"])

            total_missing_after = 0
            for qid in queries:
                total_missing_after += len(_missing_days(con, ingest_cfg.table, qid, start_day, end_day))

            total_inserted += int(round_inserted)
            all_errors.update(round_errors)
            _log(
                f"round={round_idx} inserted={round_inserted} errors={len(round_errors)} "
                f"missing_after_total={total_missing_after}"
            )

            if total_missing_after == 0:
                break

            if total_missing_after >= total_missing_before and round_inserted == 0:
                _log("no progress in this round; stopping early")
                break

            sleep_s = max(0.0, float(args.sleep_between_rounds))
            if sleep_s > 0.0 and round_idx < int(args.max_rounds):
                _log(f"sleep_between_rounds={sleep_s:.1f}s")
                time.sleep(sleep_s)

        remaining_by_query = {
            str(qid): len(_missing_days(con, ingest_cfg.table, qid, start_day, end_day)) for qid in queries
        }
        out = {
            "inserted_rows": int(total_inserted),
            "errors": all_errors,
            "start": start_day.isoformat(),
            "end": end_day.isoformat(),
            "query_count": len(queries),
            "query_ids": sorted(list(queries.keys())),
            "remaining_missing_by_query": remaining_by_query,
            "remaining_missing_total": int(sum(remaining_by_query.values())),
            "mode": str(ingest_cfg.mode).lower(),
            "dry_run": bool(args.dry_run),
        }
        print(out)
        return 0 if out["remaining_missing_total"] == 0 and not out["errors"] else 1
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
