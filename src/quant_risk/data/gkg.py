'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-11

@description: Canonical GDELT GKG raw ingestion and daily aggregation pipeline.
'''

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import hashlib
import random
import re
import time
import zipfile

import duckdb
import numpy as np
import pandas as pd


GDELT_MASTERFILELIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
_GKG_URL_TS_RE = re.compile(r"(\d{14})\.gkg\.csv\.zip$", re.IGNORECASE)


@dataclass(frozen=True)
class GkgIngestConfig:
    db_path: str
    raw_table: str = "gdelt_gkg_raw"
    daily_table: str = "news_features_daily"
    profile_name: str = "gkg_v1_light"
    profile_version: str = "2026-03-13"
    profile_mode: str = "light_daily"
    start: str = "2020-01-01"
    lookback_buffer_days: int = 14
    publication_lag_bdays: int = 1
    timeout_seconds: int = 45
    max_retries: int = 8
    retry_backoff_seconds: float = 3.0
    retry_max_sleep_seconds: float = 240.0
    request_pause_seconds: float = 1.0
    sample_interval_minutes: int = 180
    max_files_per_day: int = 8
    max_files_total: int = 0
    store_raw: bool = False
    masterfile_url: str = GDELT_MASTERFILELIST_URL
    topics: dict[str, str] | None = None
    gkg_file_urls: tuple[str, ...] = ()
    verbose: bool = True


def _log(cfg: GkgIngestConfig, message: str) -> None:
    """
    Emit a timestamped progress line if verbose mode is enabled.
    """
    if not bool(getattr(cfg, "verbose", False)):
        return
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[gkg] {ts} {message}", flush=True)


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection for GKG ingestion.
    """
    return duckdb.connect(db_path)


def init_raw_schema(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """
    Ensure the canonical GKG raw table exists.
    """
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            gkg_id TEXT NOT NULL,
            date DATE NOT NULL,
            datetime TIMESTAMP,
            source TEXT,
            url TEXT,
            language TEXT,
            tone DOUBLE,
            themes TEXT,
            locations TEXT,
            persons TEXT,
            organizations TEXT,
            query_id TEXT NOT NULL,
            source_file TEXT,
            inserted_at TIMESTAMP,
            PRIMARY KEY (gkg_id, query_id)
        )
        """
    )


def init_daily_schema(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """
    Ensure the canonical daily news-feature table exists.
    """
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            date DATE NOT NULL,
            query_id TEXT NOT NULL,
            news_count BIGINT,
            tone_mean DOUBLE,
            tone_std DOUBLE,
            tone_neg_share DOUBLE,
            source TEXT,
            inserted_at TIMESTAMP,
            PRIMARY KEY (date, query_id)
        )
        """
    )


def init_ingest_runs_schema(
    con: duckdb.DuckDBPyConnection,
    table: str = "gkg_ingest_runs",
) -> None:
    """
    Ensure the GKG ingest manifest table exists.
    """
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            run_ts TIMESTAMP NOT NULL,
            profile_name TEXT,
            profile_version TEXT,
            profile_mode TEXT,
            start_date DATE,
            end_date DATE,
            topic_ids TEXT,
            topics_hash TEXT,
            store_raw BOOLEAN,
            sample_interval_minutes INTEGER,
            max_files_per_day INTEGER,
            max_files_total INTEGER,
            publication_lag_bdays INTEGER,
            request_pause_seconds DOUBLE,
            timeout_seconds INTEGER,
            max_retries INTEGER,
            force_refresh_existing BOOLEAN,
            raw_table TEXT,
            daily_table TEXT,
            inserted_raw_rows BIGINT,
            inserted_daily_rows BIGINT,
            error_count INTEGER
        )
        """
    )


def _parse_date(value: str | None) -> date:
    """
    Parse YYYY-MM-DD into a date object.
    """
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _fetch_url_bytes(cfg: GkgIngestConfig, uri: str) -> bytes:
    """
    Fetch URI bytes with structured retry/backoff for transient failures.
    """
    max_retries = max(0, int(cfg.max_retries))
    base_sleep = max(0.0, float(cfg.retry_backoff_seconds))
    max_sleep = max(base_sleep, float(cfg.retry_max_sleep_seconds))

    for attempt in range(max_retries + 1):
        try:
            with urlopen(uri, timeout=int(cfg.timeout_seconds)) as resp:
                payload = resp.read()
            pause = float(cfg.request_pause_seconds)
            if pause > 0.0:
                time.sleep(pause)
            return payload
        except HTTPError as exc:
            code = int(getattr(exc, "code", 0) or 0)
            retryable = code == 429 or 500 <= code < 600
            if not retryable or attempt >= max_retries:
                raise
            retry_after = 0.0
            hdr = getattr(exc, "headers", None)
            if hdr is not None:
                ra = hdr.get("Retry-After")
                if ra is not None:
                    try:
                        retry_after = float(ra)
                    except Exception:
                        retry_after = 0.0
            exp_sleep = base_sleep * (2**attempt)
            jitter = random.uniform(0.0, max(base_sleep * 0.25, 0.01))
            sleep_for = min(max_sleep, max(retry_after, exp_sleep + jitter))
            _log(
                cfg,
                f"request retry uri={uri} http={code} "
                f"attempt={attempt+1}/{max_retries+1} sleep={sleep_for:.2f}s",
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
                f"request retry uri={uri} network_error "
                f"attempt={attempt+1}/{max_retries+1} sleep={sleep_for:.2f}s",
            )
            if sleep_for > 0.0:
                time.sleep(sleep_for)

    raise RuntimeError(f"Failed to fetch URI after retries: {uri}")


def _read_uri_bytes(cfg: GkgIngestConfig, uri: str) -> bytes:
    """
    Read bytes from local file path/file:// URI or remote URI.
    """
    if uri.startswith("file://"):
        path = Path(uri[len("file://") :])
        return path.read_bytes()
    p = Path(uri)
    if p.exists():
        return p.read_bytes()
    return _fetch_url_bytes(cfg, uri)


def _extract_ts_from_gkg_url(uri: str) -> Optional[datetime]:
    """
    Extract the 14-digit timestamp from a GKG ZIP URI.
    """
    m = _GKG_URL_TS_RE.search(str(uri))
    if m is None:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def list_gkg_urls_from_master(cfg: GkgIngestConfig, start_day: date, end_day: date) -> list[str]:
    """
    List GKG zip URIs from GDELT masterfilelist within [start_day, end_day].
    """
    payload = _fetch_url_bytes(cfg, str(cfg.masterfile_url)).decode("utf-8", errors="ignore")
    out_with_ts: list[tuple[datetime, str]] = []
    sample_mins = max(0, int(cfg.sample_interval_minutes))
    for line in payload.splitlines():
        parts = line.strip().split(" ")
        if len(parts) < 3:
            continue
        uri = parts[-1].strip()
        if ".gkg.csv.zip" not in uri.lower():
            continue
        ts = _extract_ts_from_gkg_url(uri)
        if ts is None:
            continue
        if sample_mins > 0:
            mins_of_day = int(ts.hour) * 60 + int(ts.minute)
            if mins_of_day % sample_mins != 0:
                continue
        d = ts.date()
        if start_day <= d <= end_day:
            out_with_ts.append((ts, uri))

    out_with_ts = sorted(set(out_with_ts), key=lambda x: x[0])
    if int(cfg.max_files_per_day) > 0:
        max_per_day = int(cfg.max_files_per_day)
        by_day_count: dict[date, int] = {}
        filtered: list[tuple[datetime, str]] = []
        for ts, uri in out_with_ts:
            d = ts.date()
            n = by_day_count.get(d, 0)
            if n >= max_per_day:
                continue
            by_day_count[d] = n + 1
            filtered.append((ts, uri))
        out_with_ts = filtered

    uris = [u for _, u in out_with_ts]
    if int(cfg.max_files_total) > 0:
        uris = uris[: int(cfg.max_files_total)]
    return uris


def _parse_tone(v2tone: str | None) -> float:
    """
    Parse tone from V2Tone string (first comma-separated value).
    """
    if v2tone is None:
        return np.nan
    head = str(v2tone).split(",", 1)[0].strip()
    if head == "":
        return np.nan
    try:
        return float(head)
    except ValueError:
        return np.nan


def _parse_language(translation_info: str | None) -> str | None:
    """
    Parse language token from TranslationInfo when available.
    """
    if translation_info is None:
        return None
    txt = str(translation_info).strip()
    if txt == "":
        return None
    m = re.search(r"\b([A-Za-z]{2,3})\b", txt)
    return m.group(1).lower() if m else None


def _empty_raw_frame() -> pd.DataFrame:
    """
    Return an empty normalized raw-frame schema.
    """
    return pd.DataFrame(
        columns=[
            "gkg_id",
            "date",
            "datetime",
            "source",
            "url",
            "language",
            "tone",
            "themes",
            "locations",
            "persons",
            "organizations",
            "source_file",
        ]
    )


def _load_gkg_zip_records(cfg: GkgIngestConfig, uri: str) -> pd.DataFrame:
    """
    Load and normalize a GKG zip file into canonical raw columns.
    """
    payload = _read_uri_bytes(cfg, uri)
    if not payload:
        return _empty_raw_frame()

    try:
        with zipfile.ZipFile(BytesIO(payload)) as zf:
            names = zf.namelist()
            if not names:
                return _empty_raw_frame()
            with zf.open(names[0]) as fh:
                content = fh.read().decode("utf-8", errors="ignore")
    except zipfile.BadZipFile:
        content = payload.decode("utf-8", errors="ignore")

    if content.strip() == "":
        return _empty_raw_frame()

    try:
        raw = pd.read_csv(
            StringIO(content),
            sep="\t",
            header=None,
            dtype=str,
            engine="python",
            on_bad_lines="skip",
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return _empty_raw_frame()

    if raw.empty:
        return _empty_raw_frame()

    def _col(idx: int) -> pd.Series:
        if idx < raw.shape[1]:
            return raw.iloc[:, idx].astype(str)
        return pd.Series([""] * len(raw), index=raw.index, dtype=str)

    record_id = _col(0).str.strip()
    dt_raw = _col(1).str.strip()
    source = _col(3).str.strip()
    url = _col(4).str.strip()

    themes = _col(8).str.strip()
    if themes.eq("").all():
        themes = _col(7).str.strip()

    locations = _col(10).str.strip()
    if locations.eq("").all():
        locations = _col(9).str.strip()

    persons = _col(12).str.strip()
    if persons.eq("").all():
        persons = _col(11).str.strip()

    organizations = _col(14).str.strip()
    if organizations.eq("").all():
        organizations = _col(13).str.strip()

    tone_v2 = _col(15)
    translation_info = _col(25)

    dt = pd.to_datetime(dt_raw, format="%Y%m%d%H%M%S", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(dt_raw, format="%Y%m%d", errors="coerce")

    out = pd.DataFrame(
        {
            "gkg_id": record_id,
            "date": dt.dt.date,
            "datetime": dt,
            "source": source.replace("", np.nan),
            "url": url.replace("", np.nan),
            "language": translation_info.map(_parse_language),
            "tone": tone_v2.map(_parse_tone),
            "themes": themes.replace("", np.nan),
            "locations": locations.replace("", np.nan),
            "persons": persons.replace("", np.nan),
            "organizations": organizations.replace("", np.nan),
            "source_file": uri,
        }
    )

    fallback_key = (
        out["url"].fillna("")
        + "|"
        + out["datetime"].astype(str)
        + "|"
        + out["source"].fillna("")
        + "|"
        + out["themes"].fillna("")
    )
    fallback_hash = fallback_key.map(lambda s: hashlib.sha1(str(s).encode("utf-8")).hexdigest())
    out["gkg_id"] = out["gkg_id"].replace("", np.nan).fillna(fallback_hash)

    out = out.dropna(subset=["date", "gkg_id"]).reset_index(drop=True)
    return out


def _extract_query_terms(query: str) -> list[str]:
    """
    Extract simple keyword terms/phrases from a boolean-like query string.
    """
    txt = str(query or "").strip()
    if txt == "":
        return []

    phrases = [p.strip().lower() for p in re.findall(r'"([^\"]+)"', txt) if p.strip()]
    txt_wo_phrases = re.sub(r'"[^\"]+"', " ", txt)
    tokens = [
        t.strip().lower()
        for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-/\.]*", txt_wo_phrases)
        if t.strip()
    ]
    stop = {"or", "and", "not", "near", "within"}
    tokens = [t for t in tokens if t not in stop]
    return list(dict.fromkeys([*phrases, *tokens]))


def _topic_fingerprint(topics: dict[str, str]) -> str:
    """
    Build a stable fingerprint for the active topic map.
    """
    items = [f"{str(k).strip()}={str(v).strip()}" for k, v in sorted((topics or {}).items())]
    return hashlib.sha1("|".join(items).encode("utf-8")).hexdigest()


def _expand_records_with_topics(raw_df: pd.DataFrame, topics: dict[str, str]) -> pd.DataFrame:
    """
    Expand normalized raw records into one row per matched query_id.
    """
    if raw_df.empty:
        return pd.DataFrame(columns=[*raw_df.columns.tolist(), "query_id"])

    if not topics:
        out = raw_df.copy()
        out["query_id"] = "default"
        return out

    haystack = (
        raw_df[["source", "url", "themes", "locations", "persons", "organizations"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )

    matched_frames: list[pd.DataFrame] = []
    for topic_id, query in topics.items():
        qid = str(topic_id).strip()
        if qid == "":
            continue
        terms = _extract_query_terms(str(query))
        if not terms:
            continue
        mask = pd.Series(False, index=raw_df.index)
        for term in terms:
            mask = mask | haystack.str.contains(re.escape(term), na=False)
        if not bool(mask.any()):
            continue
        sub = raw_df.loc[mask].copy()
        sub["query_id"] = qid
        matched_frames.append(sub)

    if not matched_frames:
        return pd.DataFrame(columns=[*raw_df.columns.tolist(), "query_id"])

    out = pd.concat(matched_frames, ignore_index=True)
    return out.drop_duplicates(subset=["gkg_id", "query_id"]).reset_index(drop=True)


def fetch_gkg_raw_records(
    cfg: GkgIngestConfig,
    *,
    start_day: date,
    end_day: date,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Fetch and normalize raw GKG records for [start_day, end_day].
    """
    errors: dict[str, str] = {}
    uris: list[str]
    if cfg.gkg_file_urls:
        uris = [u for u in cfg.gkg_file_urls]
    else:
        uris = list_gkg_urls_from_master(cfg, start_day, end_day)

    _log(cfg, f"raw fetch start files={len(uris)} range={start_day}..{end_day}")
    frames: list[pd.DataFrame] = []
    for i, uri in enumerate(uris, start=1):
        ts = _extract_ts_from_gkg_url(uri)
        if ts is not None and not (start_day <= ts.date() <= end_day):
            continue
        try:
            _log(cfg, f"raw file start {i}/{len(uris)} {uri}")
            frame = _load_gkg_zip_records(cfg, uri)
            if not frame.empty:
                frames.append(frame)
            _log(cfg, f"raw file ok {i}/{len(uris)} rows={len(frame)}")
        except Exception as exc:
            errors[str(uri)] = str(exc)
            _log(cfg, f"raw file error {i}/{len(uris)} {uri} err={exc}")

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset=["gkg_id"]).reset_index(drop=True)
    else:
        out = _empty_raw_frame()

    return out, errors


def upsert_gkg_raw(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
) -> int:
    """
    Upsert normalized raw GKG records into the canonical raw table.
    """
    if df is None or df.empty:
        return 0

    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"]).dt.date
    df2["datetime"] = pd.to_datetime(df2["datetime"], errors="coerce")
    df2["tone"] = pd.to_numeric(df2["tone"], errors="coerce")
    df2["query_id"] = df2["query_id"].astype(str)
    df2["inserted_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

    cols = [
        "gkg_id",
        "date",
        "datetime",
        "source",
        "url",
        "language",
        "tone",
        "themes",
        "locations",
        "persons",
        "organizations",
        "query_id",
        "source_file",
        "inserted_at",
    ]
    for c in cols:
        if c not in df2.columns:
            df2[c] = None

    con.register("tmp_gkg_raw", df2[cols])
    con.execute(
        f"""
        INSERT INTO {table}
            (gkg_id, date, datetime, source, url, language, tone, themes, locations, persons, organizations, query_id, source_file, inserted_at)
        SELECT
            gkg_id, date, datetime, source, url, language, tone, themes, locations, persons, organizations, query_id, source_file, inserted_at
        FROM tmp_gkg_raw
        ON CONFLICT (gkg_id, query_id) DO UPDATE SET
            date = excluded.date,
            datetime = excluded.datetime,
            source = excluded.source,
            url = excluded.url,
            language = excluded.language,
            tone = excluded.tone,
            themes = excluded.themes,
            locations = excluded.locations,
            persons = excluded.persons,
            organizations = excluded.organizations,
            source_file = excluded.source_file,
            inserted_at = excluded.inserted_at
        """
    )
    return int(len(df2))


def aggregate_daily_from_expanded(expanded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate expanded topic-tagged records to daily news features.
    """
    if expanded_df is None or expanded_df.empty:
        return pd.DataFrame(columns=["date", "query_id", "news_count", "tone_mean", "tone_std", "tone_neg_share"])

    df = expanded_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["tone"] = pd.to_numeric(df["tone"], errors="coerce")
    df["query_id"] = df["query_id"].astype(str)
    df = df.dropna(subset=["date", "query_id"])
    if df.empty:
        return pd.DataFrame(columns=["date", "query_id", "news_count", "tone_mean", "tone_std", "tone_neg_share"])

    out = (
        df.groupby(["date", "query_id"], as_index=False)
        .agg(
            news_count=("gkg_id", "count"),
            tone_mean=("tone", "mean"),
            tone_std=("tone", "std"),
            tone_neg_share=("tone", lambda s: float((pd.to_numeric(s, errors="coerce").dropna() < 0.0).mean()) if pd.to_numeric(s, errors="coerce").notna().any() else np.nan),
        )
        .sort_values(["date", "query_id"])
        .reset_index(drop=True)
    )
    return out


def upsert_daily_features(
    con: duckdb.DuckDBPyConnection,
    table: str,
    daily_df: pd.DataFrame,
    *,
    source: str = "gdelt_gkg_raw",
) -> int:
    """
    Upsert daily news feature rows into canonical daily table.
    """
    if daily_df is None or daily_df.empty:
        return 0

    df2 = daily_df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.date
    df2["query_id"] = df2["query_id"].astype(str)
    for c in ("news_count", "tone_mean", "tone_std", "tone_neg_share"):
        if c not in df2.columns:
            df2[c] = np.nan
    df2["inserted_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    df2["source"] = str(source)
    con.register("tmp_news_daily", df2)
    con.execute(
        f"""
        INSERT INTO {table}
            (date, query_id, news_count, tone_mean, tone_std, tone_neg_share, source, inserted_at)
        SELECT
            date, query_id, news_count, tone_mean, tone_std, tone_neg_share, source, inserted_at
        FROM tmp_news_daily
        ON CONFLICT (date, query_id) DO UPDATE SET
            news_count = excluded.news_count,
            tone_mean = excluded.tone_mean,
            tone_std = excluded.tone_std,
            tone_neg_share = excluded.tone_neg_share,
            source = excluded.source,
            inserted_at = excluded.inserted_at
        """
    )
    return int(len(df2))


def rebuild_daily_from_raw(
    con: duckdb.DuckDBPyConnection,
    *,
    raw_table: str,
    daily_table: str,
    start_day: date,
    end_day: date,
    query_ids: list[str] | None = None,
    source: str = "gdelt_gkg_raw",
) -> int:
    """
    Re-aggregate daily news features from raw records for a date range.
    """
    q_filter = ""
    params: list[object] = [start_day, end_day]
    if query_ids:
        q_filter = f" AND query_id IN ({','.join(['?'] * len(query_ids))})"
        params.extend(query_ids)

    daily = con.execute(
        f"""
        SELECT
            date,
            query_id,
            COUNT(*)::BIGINT AS news_count,
            AVG(tone) AS tone_mean,
            STDDEV_SAMP(tone) AS tone_std,
            AVG(CASE WHEN tone IS NOT NULL AND tone < 0 THEN 1.0 WHEN tone IS NOT NULL THEN 0.0 ELSE NULL END) AS tone_neg_share
        FROM {raw_table}
        WHERE date BETWEEN ? AND ?
          {q_filter}
        GROUP BY 1, 2
        ORDER BY 1, 2
        """,
        params,
    ).df()

    if daily.empty:
        return 0

    return upsert_daily_features(con, daily_table, daily, source=source)


def get_last_daily_date(con: duckdb.DuckDBPyConnection, table: str) -> Optional[date]:
    """
    Return max(date) in the daily news table.
    """
    row = con.execute(f"SELECT MAX(date) FROM {table}").fetchone()
    return row[0] if row and row[0] is not None else None


def _resolve_topics(cfg: GkgIngestConfig, topic_ids: list[str] | None) -> dict[str, str]:
    """
    Resolve configured topic map and optional topic-id filter.
    """
    topics = {str(k).strip(): str(v).strip() for k, v in (cfg.topics or {}).items() if str(k).strip() and str(v).strip()}
    if not topics:
        return {"default": "finance"}
    if not topic_ids:
        return topics
    allow = {str(t).strip() for t in topic_ids if str(t).strip()}
    out = {k: v for k, v in topics.items() if k in allow}
    return out


def refresh_gkg(
    cfg: GkgIngestConfig,
    *,
    start: str | None = None,
    end: str | None = None,
    topic_ids: list[str] | None = None,
    force_refresh_existing: bool = False,
) -> dict:
    """
    Ingest GKG raw records and rebuild daily features for the requested range.
    """
    con = connect(cfg.db_path)
    try:
        if bool(cfg.store_raw):
            init_raw_schema(con, cfg.raw_table)
        init_daily_schema(con, cfg.daily_table)
        init_ingest_runs_schema(con)

        start_day = _parse_date(start or cfg.start)
        end_day = _parse_date(end) if end else datetime.now(timezone.utc).date() - timedelta(days=1)
        if end_day < start_day:
            raise ValueError("end date must be >= start date")

        if not force_refresh_existing and start is None:
            last = get_last_daily_date(con, cfg.daily_table)
            if last is not None:
                buffered = last - timedelta(days=max(0, int(cfg.lookback_buffer_days) - 1))
                start_day = max(start_day, buffered)

        topics = _resolve_topics(cfg, topic_ids)
        _log(
            cfg,
            f"refresh start range={start_day}..{end_day} topics={sorted(topics.keys())} "
            f"force_refresh_existing={bool(force_refresh_existing)} store_raw={bool(cfg.store_raw)} "
            f"sample_interval_minutes={int(cfg.sample_interval_minutes)} "
            f"max_files_per_day={int(cfg.max_files_per_day)} max_files_total={int(cfg.max_files_total)}",
        )

        raw_df, errors = fetch_gkg_raw_records(cfg, start_day=start_day, end_day=end_day)
        expanded = _expand_records_with_topics(raw_df, topics)
        if bool(cfg.store_raw):
            inserted_raw = upsert_gkg_raw(con, cfg.raw_table, expanded)
            inserted_daily = rebuild_daily_from_raw(
                con,
                raw_table=cfg.raw_table,
                daily_table=cfg.daily_table,
                start_day=start_day,
                end_day=end_day,
                query_ids=sorted(topics.keys()),
                source="gdelt_gkg_raw",
            )
        else:
            inserted_raw = 0
            daily_direct = aggregate_daily_from_expanded(expanded)
            inserted_daily = upsert_daily_features(
                con,
                cfg.daily_table,
                daily_direct,
                source="gdelt_gkg_light",
            )

        out = {
            "inserted_raw_rows": int(inserted_raw),
            "inserted_daily_rows": int(inserted_daily),
            "errors": errors,
            "profile_name": str(cfg.profile_name),
            "profile_version": str(cfg.profile_version),
            "profile_mode": str(cfg.profile_mode),
            "start": start_day.isoformat(),
            "end": end_day.isoformat(),
            "topic_count": len(topics),
            "topic_ids": sorted(topics.keys()),
            "topics_hash": _topic_fingerprint(topics),
            "store_raw": bool(cfg.store_raw),
            "publication_lag_bdays": int(cfg.publication_lag_bdays),
            "request_pause_seconds": float(cfg.request_pause_seconds),
            "sample_interval_minutes": int(cfg.sample_interval_minutes),
            "max_files_per_day": int(cfg.max_files_per_day),
            "max_files_total": int(cfg.max_files_total),
            "raw_table": cfg.raw_table,
            "daily_table": cfg.daily_table,
        }
        con.execute(
            """
            INSERT INTO gkg_ingest_runs (
                run_ts,
                profile_name,
                profile_version,
                profile_mode,
                start_date,
                end_date,
                topic_ids,
                topics_hash,
                store_raw,
                sample_interval_minutes,
                max_files_per_day,
                max_files_total,
                publication_lag_bdays,
                request_pause_seconds,
                timeout_seconds,
                max_retries,
                force_refresh_existing,
                raw_table,
                daily_table,
                inserted_raw_rows,
                inserted_daily_rows,
                error_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                datetime.now(timezone.utc).replace(tzinfo=None),
                str(cfg.profile_name),
                str(cfg.profile_version),
                str(cfg.profile_mode),
                start_day,
                end_day,
                "|".join(sorted(topics.keys())),
                _topic_fingerprint(topics),
                bool(cfg.store_raw),
                int(cfg.sample_interval_minutes),
                int(cfg.max_files_per_day),
                int(cfg.max_files_total),
                int(cfg.publication_lag_bdays),
                float(cfg.request_pause_seconds),
                int(cfg.timeout_seconds),
                int(cfg.max_retries),
                bool(force_refresh_existing),
                str(cfg.raw_table),
                str(cfg.daily_table),
                int(inserted_raw),
                int(inserted_daily),
                int(len(errors)),
            ],
        )
        _log(
            cfg,
            f"refresh done profile={cfg.profile_name}:{cfg.profile_version} "
            f"inserted_raw={inserted_raw} inserted_daily={inserted_daily} errors={len(errors)}",
        )
        return out
    finally:
        con.close()
