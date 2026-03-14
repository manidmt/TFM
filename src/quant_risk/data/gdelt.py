'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-08

@description: Module to fetch and aggregate daily GDELT news signals.
'''

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from typing import Optional
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import time
import random

import duckdb
import numpy as np
import pandas as pd


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass(frozen=True)
class GdeltIngestConfig:
    db_path: str
    table: str = "gdelt_gkg_daily"
    start: str = "2018-01-01"
    lookback_buffer_days: int = 14
    publication_lag_bdays: int = 1
    mode: str = "timeline"  # timeline | artlist
    keep_artlist_sample: bool = False
    query: str = "(finance OR market OR stocks OR bitcoin OR treasury)"
    queries: dict[str, str] | None = None
    query_candidates: dict[str, list[str]] | None = None
    query_dead_min_nonzero_rate: float = 0.01
    query_dead_min_avg_news_count: float = 1.0
    max_records_per_day: int = 250
    timeout_seconds: int = 30
    timeline_chunk_days: int = 30
    max_retries: int = 6
    retry_backoff_seconds: float = 2.0
    retry_max_sleep_seconds: float = 120.0
    request_pause_seconds: float = 0.25
    verbose: bool = True


def _log(cfg: GdeltIngestConfig, message: str) -> None:
    """
    Emit a timestamped progress log line if verbose mode is enabled.
    """
    if not bool(getattr(cfg, "verbose", False)):
        return
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[gdelt] {ts} {message}", flush=True)


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection for GDELT ingestion.
    """
    return duckdb.connect(db_path)


def init_schema(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """
    Ensure the target daily GDELT table exists with the expected schema.
    """
    table_exists = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema='main' AND table_name=?
        """,
        [table],
    ).fetchone()[0] == 1

    if not table_exists:
        con.execute(
            f"""
            CREATE TABLE {table} (
                date DATE NOT NULL,
                query_id TEXT NOT NULL,
                news_count BIGINT,
                tone_mean DOUBLE,
                tone_std DOUBLE,
                tone_neg_share DOUBLE,
                source TEXT,
                inserted_at TIMESTAMP,
                PRIMARY KEY (date, query_id)
            );
            """
        )
        return

    cols = [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
    if "query_id" in cols:
        return

    tmp_table = f"{table}__migrating"
    con.execute(
        f"""
        CREATE TABLE {tmp_table} (
            date DATE NOT NULL,
            query_id TEXT NOT NULL,
            news_count BIGINT,
            tone_mean DOUBLE,
            tone_std DOUBLE,
            tone_neg_share DOUBLE,
            source TEXT,
            inserted_at TIMESTAMP,
            PRIMARY KEY (date, query_id)
        );
        """
    )
    con.execute(
        f"""
        INSERT INTO {tmp_table} (date, query_id, news_count, tone_mean, tone_std, tone_neg_share, source, inserted_at)
        SELECT date, 'default' AS query_id, news_count, tone_mean, tone_std, tone_neg_share, source, inserted_at
        FROM {table}
        """
    )
    con.execute(f"DROP TABLE {table}")
    con.execute(f"ALTER TABLE {tmp_table} RENAME TO {table}")


def get_last_date(con: duckdb.DuckDBPyConnection, table: str) -> Optional[date]:
    """
    Return the latest stored date in the daily GDELT table.
    """
    row = con.execute(f"SELECT MAX(date) FROM {table}").fetchone()
    return row[0] if row and row[0] is not None else None


def _day_to_api_start(day: date) -> str:
    """
    Convert a date to GDELT API startdatetime format (YYYYMMDD000000).
    """
    return day.strftime("%Y%m%d") + "000000"


def _day_to_api_end(day: date) -> str:
    """
    Convert a date to GDELT API enddatetime format (YYYYMMDD235959).
    """
    return day.strftime("%Y%m%d") + "235959"


def _iter_days(start_day: date, end_day: date) -> list[date]:
    """
    Build an inclusive list of daily dates between start_day and end_day.
    """
    n_days = (end_day - start_day).days
    if n_days < 0:
        return []
    return [start_day + timedelta(days=i) for i in range(n_days + 1)]


def _iter_date_chunks(start_day: date, end_day: date, chunk_days: int) -> list[tuple[date, date]]:
    """
    Build inclusive date chunks of size chunk_days.
    """
    if chunk_days <= 0:
        return [(start_day, end_day)]
    chunks: list[tuple[date, date]] = []
    cur = start_day
    while cur <= end_day:
        nxt = min(cur + timedelta(days=chunk_days - 1), end_day)
        chunks.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return chunks


def _load_csv_from_url(url: str, timeout_seconds: int) -> pd.DataFrame:
    """
    Download a CSV payload from URL and parse it as a pandas DataFrame.
    """
    with urlopen(url, timeout=timeout_seconds) as resp:
        payload = resp.read().decode("utf-8", errors="ignore")
    if not payload.strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(StringIO(payload))
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        # GDELT can return an empty/degenerate payload for windows with no matches.
        return pd.DataFrame()


def _fetch_doc_mode_csv(
    cfg: GdeltIngestConfig,
    mode: str,
    start_day: date,
    end_day: date,
    query: str,
    request_label: str | None = None,
) -> pd.DataFrame:
    """
    Fetch a CSV response from GDELT Doc API for a date range and mode.
    """
    params = {
        "query": str(query),
        "mode": mode,
        "format": "CSV",
        "startdatetime": _day_to_api_start(start_day),
        "enddatetime": _day_to_api_end(end_day),
    }
    if str(mode).lower() == "artlist":
        params["maxrecords"] = int(cfg.max_records_per_day)

    url = f"{GDELT_DOC_API}?{urlencode(params)}"
    max_retries = max(0, int(cfg.max_retries))
    base_sleep = max(0.0, float(cfg.retry_backoff_seconds))
    max_sleep = max(base_sleep, float(cfg.retry_max_sleep_seconds))

    for attempt in range(max_retries + 1):
        try:
            if attempt == 0:
                _log(
                    cfg,
                    f"request start mode={mode} range={start_day}..{end_day} label={request_label or '-'}",
                )
            out = _load_csv_from_url(url, timeout_seconds=int(cfg.timeout_seconds))
            _log(
                cfg,
                f"request ok mode={mode} range={start_day}..{end_day} label={request_label or '-'} rows={len(out)} attempt={attempt+1}",
            )
            pause = float(cfg.request_pause_seconds)
            if pause > 0.0:
                time.sleep(pause)
            return out
        except HTTPError as exc:
            code = int(getattr(exc, "code", 0) or 0)
            retryable = code == 429 or 500 <= code < 600
            if not retryable or attempt >= max_retries:
                raise

            retry_after = 0.0
            try:
                hdr = getattr(exc, "headers", None)
                if hdr is not None:
                    ra = hdr.get("Retry-After")
                    if ra is not None:
                        retry_after = float(ra)
            except Exception:
                retry_after = 0.0

            exp_sleep = base_sleep * (2**attempt)
            jitter = random.uniform(0.0, max(base_sleep * 0.25, 0.01))
            sleep_for = min(max_sleep, max(retry_after, exp_sleep + jitter))
            _log(
                cfg,
                f"request retry mode={mode} range={start_day}..{end_day} label={request_label or '-'} "
                f"http={code} attempt={attempt+1}/{max_retries+1} sleep={sleep_for:.2f}s",
            )
            if sleep_for > 0.0:
                time.sleep(sleep_for)
        except URLError:
            if attempt >= max_retries:
                raise
            exp_sleep = base_sleep * (2**attempt)
            jitter = random.uniform(0.0, max(base_sleep * 0.25, 0.01))
            sleep_for = min(max_sleep, exp_sleep + jitter)
            _log(
                cfg,
                f"request retry mode={mode} range={start_day}..{end_day} label={request_label or '-'} "
                f"network_error attempt={attempt+1}/{max_retries+1} sleep={sleep_for:.2f}s",
            )
            if sleep_for > 0.0:
                time.sleep(sleep_for)

    return pd.DataFrame()


def _normalize_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw GDELT records to a minimal article schema (date, tone).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "tone"])

    out = df.copy()
    lower_to_original = {str(c).strip().lower(): c for c in out.columns}

    date_col = None
    for cand in ("seendate", "date", "datetime", "day"):
        if cand in lower_to_original:
            date_col = lower_to_original[cand]
            break
    if date_col is None:
        return pd.DataFrame(columns=["date", "tone"])

    tone_col = None
    for cand in ("tone", "v2tone"):
        if cand in lower_to_original:
            tone_col = lower_to_original[cand]
            break

    norm = pd.DataFrame()
    norm["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.date

    if tone_col is not None:
        tone_raw = out[tone_col].astype(str).str.split(",").str[0]
        norm["tone"] = pd.to_numeric(tone_raw, errors="coerce")
    else:
        norm["tone"] = np.nan

    norm = norm.dropna(subset=["date"]).reset_index(drop=True)
    return norm


def fetch_day_articles(cfg: GdeltIngestConfig, day: date, query: str | None = None) -> pd.DataFrame:
    """
    Fetch daily article-level records from GDELT Doc API for one calendar day.
    """
    raw = _fetch_doc_mode_csv(
        cfg,
        mode="ArtList",
        start_day=day,
        end_day=day,
        query=str(query or cfg.query),
        request_label="artlist-day",
    )
    return _normalize_articles(raw)


def aggregate_daily(articles: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article-level rows into daily news_count and tone statistics.
    """
    if articles is None or articles.empty:
        return pd.DataFrame(columns=["date", "news_count", "tone_mean", "tone_std", "tone_neg_share"])

    frame = articles.copy()
    frame["tone"] = pd.to_numeric(frame["tone"], errors="coerce")
    grouped = frame.groupby("date")
    out = grouped["tone"].agg(tone_mean="mean", tone_std="std").reset_index()
    out["news_count"] = grouped.size().to_numpy()

    def _neg_share(s: pd.Series) -> float:
        v = pd.to_numeric(s, errors="coerce").dropna()
        if v.empty:
            return np.nan
        return float((v < 0.0).mean())

    neg = grouped["tone"].apply(_neg_share).reset_index()
    neg.columns = ["date", "tone_neg_share"]
    out = out.merge(neg, on="date", how="left")
    out["tone_std"] = out["tone_std"].fillna(0.0)
    cols = ["date", "news_count", "tone_mean", "tone_std", "tone_neg_share"]
    return out[cols].sort_values("date").reset_index(drop=True)


def _normalize_timeline_metric(df: pd.DataFrame, value_candidates: tuple[str, ...]) -> pd.DataFrame:
    """
    Normalize timeline CSV to (date, value) using robust date/value column detection.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "value"])

    out = df.copy()
    lower_to_original = {str(c).strip().lower(): c for c in out.columns}

    date_col = None
    for cand in ("date", "datetime", "day", "timeslot", "time"):
        if cand in lower_to_original:
            date_col = lower_to_original[cand]
            break

    if date_col is None:
        for c in out.columns:
            parsed = pd.to_datetime(out[c], errors="coerce")
            if parsed.notna().any():
                date_col = c
                break
    if date_col is None:
        return pd.DataFrame(columns=["date", "value"])

    value_col = None
    for cand in value_candidates:
        k = str(cand).strip().lower()
        if k in lower_to_original:
            value_col = lower_to_original[k]
            break

    if value_col is None:
        for c in out.columns:
            if c == date_col:
                continue
            as_num = pd.to_numeric(out[c], errors="coerce")
            if as_num.notna().any():
                value_col = c
                break
    if value_col is None:
        return pd.DataFrame(columns=["date", "value"])

    norm = pd.DataFrame()
    norm["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.date
    norm["value"] = pd.to_numeric(out[value_col], errors="coerce")
    norm = norm.dropna(subset=["date"])
    return norm.reset_index(drop=True)


def fetch_timeline_volraw(
    cfg: GdeltIngestConfig,
    start_day: date,
    end_day: date,
    query: str | None = None,
) -> pd.DataFrame:
    """
    Fetch timeline volume aggregates and map them to daily news_count.
    """
    raw = _fetch_doc_mode_csv(
        cfg,
        mode="TimelineVolRaw",
        start_day=start_day,
        end_day=end_day,
        query=str(query or cfg.query),
        request_label="timeline-volraw",
    )
    norm = _normalize_timeline_metric(
        raw,
        value_candidates=("count", "volume", "vol", "value"),
    )
    if norm.empty:
        return pd.DataFrame(columns=["date", "news_count"])

    out = norm.groupby("date", as_index=False)["value"].sum().rename(columns={"value": "news_count"})
    return out.sort_values("date").reset_index(drop=True)


def fetch_timeline_tone(
    cfg: GdeltIngestConfig,
    start_day: date,
    end_day: date,
    query: str | None = None,
) -> pd.DataFrame:
    """
    Fetch timeline tone aggregates and map them to daily tone_mean.
    """
    raw = _fetch_doc_mode_csv(
        cfg,
        mode="TimelineTone",
        start_day=start_day,
        end_day=end_day,
        query=str(query or cfg.query),
        request_label="timeline-tone",
    )
    norm = _normalize_timeline_metric(
        raw,
        value_candidates=("tone", "value", "avgtone", "meantone"),
    )
    if norm.empty:
        return pd.DataFrame(columns=["date", "tone_mean"])

    out = norm.groupby("date", as_index=False)["value"].mean().rename(columns={"value": "tone_mean"})
    return out.sort_values("date").reset_index(drop=True)


def fetch_artlist_range(cfg: GdeltIngestConfig, start_day: date, end_day: date) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Fetch ArtList samples per day for a range and return daily aggregates and per-day errors.
    """
    days = _iter_days(start_day, end_day)
    daily_rows: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    for day in days:
        try:
            articles = fetch_day_articles(cfg, day)
            daily = aggregate_daily(articles)
            if not daily.empty:
                daily_rows.append(daily)
        except Exception as exc:
            errors[day.isoformat()] = str(exc)

    if daily_rows:
        merged = pd.concat(daily_rows, ignore_index=True)
        merged = (
            merged.sort_values("date")
            .groupby("date", as_index=False)
            .agg(
                news_count=("news_count", "sum"),
                tone_mean=("tone_mean", "mean"),
                tone_std=("tone_std", "mean"),
                tone_neg_share=("tone_neg_share", "mean"),
            )
        )
    else:
        merged = pd.DataFrame(columns=["date", "news_count", "tone_mean", "tone_std", "tone_neg_share"])
    return merged, errors


def fetch_timeline_range(cfg: GdeltIngestConfig, start_day: date, end_day: date) -> pd.DataFrame:
    """
    Fetch timeline volume/tone metrics for a range and return a daily merged dataframe.
    """
    out, _ = fetch_timeline_range_for_query(cfg, start_day, end_day, query_text=str(cfg.query))
    return out


def _queries_from_cfg(cfg: GdeltIngestConfig) -> dict[str, str]:
    """
    Return configured thematic queries as {query_id: query_text}.
    """
    if cfg.queries:
        out: dict[str, str] = {}
        for k, v in cfg.queries.items():
            qid = str(k).strip()
            qtxt = str(v).strip()
            if not qid or not qtxt:
                continue
            out[qid] = qtxt
        if out:
            return out
    return {"default": str(cfg.query)}


def _query_candidates_for_id(cfg: GdeltIngestConfig, query_id: str, primary_query: str) -> list[str]:
    """
    Build ordered candidates for one query_id, primary first and deduplicated.
    """
    out: list[str] = [str(primary_query).strip()]
    candidates_cfg = cfg.query_candidates or {}
    extra = candidates_cfg.get(str(query_id), [])
    if isinstance(extra, str):
        extra = [extra]
    for q in extra:
        qtxt = str(q).strip()
        if qtxt and qtxt not in out:
            out.append(qtxt)
    return out


def _daily_signal_stats(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute simple quality stats for a fetched daily signal frame.
    """
    if df is None or df.empty or "news_count" not in df.columns:
        return {
            "rows": 0.0,
            "positive_days": 0.0,
            "nonzero_rate": 0.0,
            "avg_news_count": 0.0,
        }
    news = pd.to_numeric(df["news_count"], errors="coerce").fillna(0.0)
    rows = float(len(news))
    positive_days = float((news > 0.0).sum())
    nonzero_rate = float(positive_days / rows) if rows > 0 else 0.0
    avg_news_count = float(news.mean()) if rows > 0 else 0.0
    return {
        "rows": rows,
        "positive_days": positive_days,
        "nonzero_rate": nonzero_rate,
        "avg_news_count": avg_news_count,
    }


def _is_dead_signal(stats: dict[str, float], cfg: GdeltIngestConfig) -> bool:
    """
    Flag a daily signal as dead (all-zero or near-all-zero).
    """
    rows = float(stats.get("rows", 0.0))
    positive_days = float(stats.get("positive_days", 0.0))
    nonzero_rate = float(stats.get("nonzero_rate", 0.0))
    avg_news_count = float(stats.get("avg_news_count", 0.0))
    if rows <= 0 or positive_days <= 0:
        return True
    return (
        nonzero_rate < float(cfg.query_dead_min_nonzero_rate)
        and avg_news_count < float(cfg.query_dead_min_avg_news_count)
    )


def fetch_artlist_range_for_query(
    cfg: GdeltIngestConfig,
    start_day: date,
    end_day: date,
    query_text: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Fetch ArtList daily samples for a single query over a date range.
    """
    days = _iter_days(start_day, end_day)
    daily_rows: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    for day in days:
        try:
            articles = fetch_day_articles(cfg, day, query=query_text)
            daily = aggregate_daily(articles)
            if not daily.empty:
                daily_rows.append(daily)
        except Exception as exc:
            errors[day.isoformat()] = str(exc)

    if daily_rows:
        merged = pd.concat(daily_rows, ignore_index=True)
        merged = (
            merged.sort_values("date")
            .groupby("date", as_index=False)
            .agg(
                news_count=("news_count", "sum"),
                tone_mean=("tone_mean", "mean"),
                tone_std=("tone_std", "mean"),
                tone_neg_share=("tone_neg_share", "mean"),
            )
        )
    else:
        merged = pd.DataFrame(columns=["date", "news_count", "tone_mean", "tone_std", "tone_neg_share"])
    return merged, errors


def fetch_timeline_range_for_query(
    cfg: GdeltIngestConfig,
    start_day: date,
    end_day: date,
    query_text: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Fetch timeline aggregates for a single query over a date range.
    """
    errors: dict[str, str] = {}
    chunks = _iter_date_chunks(start_day, end_day, int(cfg.timeline_chunk_days))
    chunk_rows: list[pd.DataFrame] = []
    _log(
        cfg,
        f"query start mode=timeline chunks={len(chunks)} range={start_day}..{end_day} "
        f"query='{query_text[:80]}'",
    )

    for idx, (c_start, c_end) in enumerate(chunks, start=1):
        chunk_key = f"{c_start.isoformat()}..{c_end.isoformat()}"
        try:
            _log(cfg, f"chunk start {idx}/{len(chunks)} {chunk_key}")
            vol = fetch_timeline_volraw(cfg, c_start, c_end, query=query_text)
            tone = fetch_timeline_tone(cfg, c_start, c_end, query=query_text)
            chunk = vol.merge(tone, on="date", how="outer").sort_values("date").reset_index(drop=True)
            if not chunk.empty:
                chunk_rows.append(chunk)
            _log(
                cfg,
                f"chunk ok {idx}/{len(chunks)} {chunk_key} rows_vol={len(vol)} rows_tone={len(tone)} rows_merged={len(chunk)}",
            )
        except Exception as exc:
            errors[chunk_key] = str(exc)
            _log(cfg, f"chunk error {idx}/{len(chunks)} {chunk_key}: {exc}")

    if chunk_rows:
        out = pd.concat(chunk_rows, ignore_index=True)
        out = (
            out.groupby("date", as_index=False)
            .agg(
                news_count=("news_count", "sum"),
                tone_mean=("tone_mean", "mean"),
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
    else:
        out = pd.DataFrame(columns=["date", "news_count", "tone_mean"])

    out["tone_std"] = np.nan
    out["tone_neg_share"] = np.nan

    if bool(cfg.keep_artlist_sample):
        art_sample, art_errors = fetch_artlist_range_for_query(cfg, start_day, end_day, query_text=query_text)
        for k, msg in art_errors.items():
            errors[f"artlist:{k}"] = msg
        if not art_sample.empty:
            sample_cols = ["date", "tone_std", "tone_neg_share"]
            art_keep = art_sample[sample_cols].drop_duplicates(subset=["date"])
            out = out.merge(art_keep, on="date", how="left", suffixes=("", "_sample"))
            out["tone_std"] = out["tone_std"].where(out["tone_std"].notna(), out["tone_std_sample"])
            out["tone_neg_share"] = out["tone_neg_share"].where(
                out["tone_neg_share"].notna(), out["tone_neg_share_sample"]
            )
            drop_cols = [c for c in ["tone_std_sample", "tone_neg_share_sample"] if c in out.columns]
            if drop_cols:
                out = out.drop(columns=drop_cols)

    cols = ["date", "news_count", "tone_mean", "tone_std", "tone_neg_share"]
    out = out[cols].sort_values("date").reset_index(drop=True)
    _log(
        cfg,
        f"query done mode=timeline range={start_day}..{end_day} rows={len(out)} errors={len(errors)}",
    )
    return out, errors


def upsert_gdelt_daily(
    con: duckdb.DuckDBPyConnection,
    table: str,
    daily_df: pd.DataFrame,
    source: str = "gdelt_doc_api",
) -> int:
    """
    Upsert daily GDELT aggregates into DuckDB by (date, query_id) primary key.
    """
    if daily_df is None or daily_df.empty:
        return 0

    df2 = daily_df.copy()
    if "query_id" not in df2.columns:
        df2["query_id"] = "default"
    df2["query_id"] = df2["query_id"].astype(str).replace("", "default")
    for c in ["news_count", "tone_mean", "tone_std", "tone_neg_share"]:
        if c not in df2.columns:
            df2[c] = np.nan
    df2["source"] = source
    df2["inserted_at"] = datetime.now(timezone.utc)
    con.register("tmp_gdelt_daily", df2)
    con.execute(
        f"""
        INSERT INTO {table} (date, query_id, news_count, tone_mean, tone_std, tone_neg_share, source, inserted_at)
        SELECT date, query_id, news_count, tone_mean, tone_std, tone_neg_share, source, inserted_at
        FROM tmp_gdelt_daily
        ON CONFLICT (date, query_id) DO UPDATE SET
            news_count = excluded.news_count,
            tone_mean = excluded.tone_mean,
            tone_std = excluded.tone_std,
            tone_neg_share = excluded.tone_neg_share,
            source = excluded.source,
            inserted_at = excluded.inserted_at
        """
    )
    return len(df2)


def refresh_gdelt(
    cfg: GdeltIngestConfig,
    start: Optional[str] = None,
    end: Optional[str] = None,
    query_ids: Optional[list[str]] = None,
) -> dict:
    """
    Run incremental ingestion from GDELT API and persist daily aggregates.
    """
    con = connect(cfg.db_path)
    try:
        init_schema(con, cfg.table)

        if start:
            start_day = pd.to_datetime(start).date()
        else:
            last = get_last_date(con, cfg.table)
            if last is not None:
                start_day = last - timedelta(days=int(cfg.lookback_buffer_days))
            else:
                start_day = pd.to_datetime(cfg.start).date()

        end_day = pd.to_datetime(end).date() if end else date.today()
        mode = str(cfg.mode).strip().lower()
        errors: dict[str, str] = {}
        queries = _queries_from_cfg(cfg)
        if query_ids:
            keep = {str(q).strip() for q in query_ids if str(q).strip()}
            queries = {k: v for k, v in queries.items() if k in keep}
            if not queries:
                raise ValueError(f"query_ids={query_ids!r} does not match configured ids.")
        merged_rows: list[pd.DataFrame] = []
        selected_query_text: dict[str, str] = {}
        _log(
            cfg,
            f"refresh start mode={mode} start={start_day} end={end_day} query_count={len(queries)}",
        )

        for query_id, query_text in queries.items():
            _log(cfg, f"query_id start id={query_id}")
            candidates = _query_candidates_for_id(cfg, query_id=str(query_id), primary_query=str(query_text))
            selected_df = pd.DataFrame()
            selected_stats = {"rows": 0.0, "positive_days": 0.0, "nonzero_rate": 0.0, "avg_news_count": 0.0}
            selected_query = candidates[0]

            for cand_idx, cand_query in enumerate(candidates, start=1):
                if mode == "timeline":
                    q_df, q_errors = fetch_timeline_range_for_query(
                        cfg,
                        start_day,
                        end_day,
                        query_text=cand_query,
                    )
                    for chunk_key, msg in q_errors.items():
                        errors[f"{query_id}:cand{cand_idx}:timeline:{chunk_key}"] = msg
                elif mode == "artlist":
                    q_df, q_errors = fetch_artlist_range_for_query(
                        cfg,
                        start_day,
                        end_day,
                        query_text=cand_query,
                    )
                    for day_key, msg in q_errors.items():
                        errors[f"{query_id}:cand{cand_idx}:{day_key}"] = msg
                else:
                    raise ValueError(f"Unsupported gdelt.mode={cfg.mode!r}. Use 'timeline' or 'artlist'.")

                stats = _daily_signal_stats(q_df)
                is_dead = _is_dead_signal(stats, cfg)
                _log(
                    cfg,
                    f"query_id={query_id} candidate={cand_idx}/{len(candidates)} "
                    f"rows={int(stats['rows'])} positive_days={int(stats['positive_days'])} "
                    f"nonzero_rate={stats['nonzero_rate']:.4f} avg_news_count={stats['avg_news_count']:.2f} dead={is_dead}",
                )

                # Prefer first viable candidate; otherwise keep best by positive_days/nonzero_rate/avg.
                better = (
                    (stats["positive_days"], stats["nonzero_rate"], stats["avg_news_count"])
                    > (
                        selected_stats["positive_days"],
                        selected_stats["nonzero_rate"],
                        selected_stats["avg_news_count"],
                    )
                )
                if better:
                    selected_df = q_df
                    selected_stats = stats
                    selected_query = cand_query

                if not is_dead:
                    selected_df = q_df
                    selected_stats = stats
                    selected_query = cand_query
                    break

            if not selected_df.empty:
                q_df = selected_df.copy()
                q_df["query_id"] = str(query_id)
                merged_rows.append(q_df)
            selected_query_text[str(query_id)] = selected_query
            _log(
                cfg,
                f"query_id done id={query_id} rows={len(selected_df)} "
                f"errors={sum(1 for k in errors if k.startswith(f'{query_id}:'))}",
            )

        if merged_rows:
            merged = pd.concat(merged_rows, ignore_index=True)
        else:
            merged = pd.DataFrame(
                columns=["date", "query_id", "news_count", "tone_mean", "tone_std", "tone_neg_share"]
            )

        inserted = upsert_gdelt_daily(con, cfg.table, merged, source="gdelt_doc_api")
        out = {
            "inserted_rows": int(inserted),
            "errors": errors,
            "start": start_day.isoformat(),
            "end": end_day.isoformat(),
            "days_queried": len(_iter_days(start_day, end_day)),
            "days_with_news": int(merged["date"].nunique()) if "date" in merged.columns else 0,
            "rows_with_query": int(merged.shape[0]),
            "query_count": len(queries),
            "query_ids": sorted(list(queries.keys())),
            "selected_query_text": selected_query_text,
            "mode": mode,
        }
        _log(
            cfg,
            f"refresh done inserted_rows={out['inserted_rows']} days_with_news={out['days_with_news']} "
            f"rows_with_query={out['rows_with_query']} errors={len(errors)}",
        )
        return out
    finally:
        con.close()
