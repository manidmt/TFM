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
from urllib.request import urlopen

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
    max_records_per_day: int = 250
    timeout_seconds: int = 30


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection for GDELT ingestion.
    """
    return duckdb.connect(db_path)


def init_schema(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """
    Ensure the target daily GDELT table exists with the expected schema.
    """
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            date DATE NOT NULL,
            news_count BIGINT,
            tone_mean DOUBLE,
            tone_std DOUBLE,
            tone_neg_share DOUBLE,
            source TEXT,
            inserted_at TIMESTAMP,
            PRIMARY KEY (date)
        );
        """
    )


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


def _load_csv_from_url(url: str, timeout_seconds: int) -> pd.DataFrame:
    """
    Download a CSV payload from URL and parse it as a pandas DataFrame.
    """
    with urlopen(url, timeout=timeout_seconds) as resp:
        payload = resp.read().decode("utf-8", errors="ignore")
    if not payload.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(payload))


def _fetch_doc_mode_csv(
    cfg: GdeltIngestConfig,
    mode: str,
    start_day: date,
    end_day: date,
) -> pd.DataFrame:
    """
    Fetch a CSV response from GDELT Doc API for a date range and mode.
    """
    params = {
        "query": cfg.query,
        "mode": mode,
        "format": "CSV",
        "startdatetime": _day_to_api_start(start_day),
        "enddatetime": _day_to_api_end(end_day),
    }
    if str(mode).lower() == "artlist":
        params["maxrecords"] = int(cfg.max_records_per_day)

    url = f"{GDELT_DOC_API}?{urlencode(params)}"
    return _load_csv_from_url(url, timeout_seconds=int(cfg.timeout_seconds))


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


def fetch_day_articles(cfg: GdeltIngestConfig, day: date) -> pd.DataFrame:
    """
    Fetch daily article-level records from GDELT Doc API for one calendar day.
    """
    raw = _fetch_doc_mode_csv(cfg, mode="ArtList", start_day=day, end_day=day)
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


def fetch_timeline_volraw(cfg: GdeltIngestConfig, start_day: date, end_day: date) -> pd.DataFrame:
    """
    Fetch timeline volume aggregates and map them to daily news_count.
    """
    raw = _fetch_doc_mode_csv(cfg, mode="TimelineVolRaw", start_day=start_day, end_day=end_day)
    norm = _normalize_timeline_metric(
        raw,
        value_candidates=("count", "volume", "vol", "value"),
    )
    if norm.empty:
        return pd.DataFrame(columns=["date", "news_count"])

    out = norm.groupby("date", as_index=False)["value"].sum().rename(columns={"value": "news_count"})
    return out.sort_values("date").reset_index(drop=True)


def fetch_timeline_tone(cfg: GdeltIngestConfig, start_day: date, end_day: date) -> pd.DataFrame:
    """
    Fetch timeline tone aggregates and map them to daily tone_mean.
    """
    raw = _fetch_doc_mode_csv(cfg, mode="TimelineTone", start_day=start_day, end_day=end_day)
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
    vol = fetch_timeline_volraw(cfg, start_day, end_day)
    tone = fetch_timeline_tone(cfg, start_day, end_day)
    out = vol.merge(tone, on="date", how="outer").sort_values("date").reset_index(drop=True)
    if out.empty:
        out = pd.DataFrame(columns=["date", "news_count", "tone_mean"])

    out["tone_std"] = np.nan
    out["tone_neg_share"] = np.nan

    if bool(cfg.keep_artlist_sample):
        art_sample, _ = fetch_artlist_range(cfg, start_day, end_day)
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
    return out[cols].sort_values("date").reset_index(drop=True)


def upsert_gdelt_daily(
    con: duckdb.DuckDBPyConnection,
    table: str,
    daily_df: pd.DataFrame,
    source: str = "gdelt_doc_api",
) -> int:
    """
    Upsert daily GDELT aggregates into DuckDB by date primary key.
    """
    if daily_df is None or daily_df.empty:
        return 0

    df2 = daily_df.copy()
    for c in ["news_count", "tone_mean", "tone_std", "tone_neg_share"]:
        if c not in df2.columns:
            df2[c] = np.nan
    df2["source"] = source
    df2["inserted_at"] = datetime.now(timezone.utc)
    con.register("tmp_gdelt_daily", df2)
    con.execute(
        f"""
        INSERT INTO {table} (date, news_count, tone_mean, tone_std, tone_neg_share, source, inserted_at)
        SELECT date, news_count, tone_mean, tone_std, tone_neg_share, source, inserted_at
        FROM tmp_gdelt_daily
        ON CONFLICT (date) DO UPDATE SET
            news_count = excluded.news_count,
            tone_mean = excluded.tone_mean,
            tone_std = excluded.tone_std,
            tone_neg_share = excluded.tone_neg_share,
            source = excluded.source,
            inserted_at = excluded.inserted_at
        """
    )
    return len(df2)


def refresh_gdelt(cfg: GdeltIngestConfig, end: Optional[str] = None) -> dict:
    """
    Run incremental ingestion from GDELT API and persist daily aggregates.
    """
    con = connect(cfg.db_path)
    try:
        init_schema(con, cfg.table)

        last = get_last_date(con, cfg.table)
        if last is not None:
            start_day = last - timedelta(days=int(cfg.lookback_buffer_days))
        else:
            start_day = pd.to_datetime(cfg.start).date()

        end_day = pd.to_datetime(end).date() if end else date.today()
        mode = str(cfg.mode).strip().lower()
        errors: dict[str, str] = {}

        if mode == "timeline":
            try:
                merged = fetch_timeline_range(cfg, start_day, end_day)
            except Exception as exc:
                errors["timeline"] = str(exc)
                merged = pd.DataFrame(columns=["date", "news_count", "tone_mean", "tone_std", "tone_neg_share"])
        elif mode == "artlist":
            merged, errors = fetch_artlist_range(cfg, start_day, end_day)
        else:
            raise ValueError(f"Unsupported gdelt.mode={cfg.mode!r}. Use 'timeline' or 'artlist'.")

        inserted = upsert_gdelt_daily(con, cfg.table, merged, source="gdelt_doc_api")
        return {
            "inserted_rows": int(inserted),
            "errors": errors,
            "start": start_day.isoformat(),
            "end": end_day.isoformat(),
            "days_queried": len(_iter_days(start_day, end_day)),
            "days_with_news": int(merged.shape[0]),
            "mode": mode,
        }
    finally:
        con.close()
