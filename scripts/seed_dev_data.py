#!/usr/bin/env python3
"""
seed_dev_data.py — insert / wipe synthetic predictions in serving.duckdb.

Usage
-----
    python scripts/seed_dev_data.py                  # seed 90 days
    python scripts/seed_dev_data.py --wipe           # delete synthetic rows
    python scripts/seed_dev_data.py --db /path/to.db # custom DB path

All synthetic rows carry  bundle_version = 'synthetic-seed'.
No quant_risk imports — only duckdb and stdlib.
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import date, datetime, timedelta, timezone

import duckdb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKER = "synthetic-seed"

ASSETS = [
    "us_equities",
    "euro_equities",
    "bitcoin",
    "long_us_treasuries",
    "short_us_treasuries",
    "gold",
]

REGIMES = ["low", "medium", "high"]

# Default transition matrix  [from_low, from_medium, from_high]
# Each row sums to 1.0.
BASE_TRANSITIONS: dict[str, list[float]] = {
    "low":    [0.70, 0.30, 0.00],
    "medium": [0.20, 0.60, 0.20],
    "high":   [0.00, 0.40, 0.60],
}

# Bitcoin: more volatile — elevated high↔high probability
BTC_TRANSITIONS: dict[str, list[float]] = {
    "low":    [0.55, 0.40, 0.05],
    "medium": [0.15, 0.50, 0.35],
    "high":   [0.00, 0.30, 0.70],
}

# Dirichlet α per regime — shapes probability vector toward that regime
DIRICHLET_ALPHA: dict[str, list[float]] = {
    "low":    [6.0, 2.0, 0.5],
    "medium": [1.5, 5.0, 1.5],
    "high":   [0.5, 2.0, 6.0],
}

DEFAULT_DB = os.environ.get(
    "QUANT_RISK_SERVING_DB_PATH",
    "/srv/quant-risk/db/serving.duckdb",
)

_DDL = """
CREATE TABLE IF NOT EXISTS predictions_daily (
    asset_id         VARCHAR  NOT NULL,
    forecast_date    DATE     NOT NULL,
    predicted_class  VARCHAR  NOT NULL,
    p_low            DOUBLE   NOT NULL,
    p_medium         DOUBLE   NOT NULL,
    p_high           DOUBLE   NOT NULL,
    bundle_version   VARCHAR  NOT NULL,
    data_cutoff_date DATE     NOT NULL,
    status           VARCHAR  NOT NULL,
    updated_at       TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (asset_id, forecast_date)
);
"""


# ---------------------------------------------------------------------------
# Stubs — implemented in Task 2
# ---------------------------------------------------------------------------

def _markov_next(current: str, transitions: dict[str, list[float]], rng: random.Random) -> str:
    """Sample next regime using the transition row for `current`."""
    weights = transitions[current]
    return rng.choices(REGIMES, weights=weights, k=1)[0]

def _dirichlet_sample(alpha: list[float], rng: random.Random) -> tuple[float, float, float]:
    """Sample a probability triple from Dirichlet(alpha) using the gamma trick."""
    gammas = [rng.gammavariate(a, 1.0) for a in alpha]
    total = sum(gammas)
    p = [g / total for g in gammas]
    return (p[0], p[1], p[2])

def _generate_rows(asset_id: str, n_days: int) -> list[dict]:
    """Generate n_days+1 rows (today-n_days … today) for one asset."""
    seed_val = sum(ord(c) for c in asset_id)
    rng = random.Random(seed_val)

    transitions = BTC_TRANSITIONS if asset_id == "bitcoin" else BASE_TRANSITIONS
    # Bitcoin starts in high; others start in low
    regime = "high" if asset_id == "bitcoin" else "low"

    today = date.today()
    now = datetime.now(timezone.utc)
    rows = []

    for offset in range(n_days, -1, -1):  # oldest → newest
        forecast_date = today - timedelta(days=offset)
        p_low, p_med, p_high = _dirichlet_sample(DIRICHLET_ALPHA[regime], rng)
        rows.append({
            "asset_id": asset_id,
            "forecast_date": forecast_date,
            "predicted_class": regime,
            "p_low": p_low,
            "p_medium": p_med,
            "p_high": p_high,
            "bundle_version": MARKER,
            "data_cutoff_date": forecast_date,
            "status": "success",
            "updated_at": now,
        })
        regime = _markov_next(regime, transitions, rng)

    return rows

def seed(db_path: str, n_days: int = 90) -> int:
    """Insert synthetic predictions. Returns number of rows upserted."""
    con = duckdb.connect(db_path)
    try:
        con.execute(_DDL)
        total = 0
        for asset_id in ASSETS:
            rows = _generate_rows(asset_id, n_days)
            for r in rows:
                con.execute(
                    "DELETE FROM predictions_daily WHERE asset_id = ? AND forecast_date = ?",
                    [r["asset_id"], r["forecast_date"]],
                )
                con.execute(
                    """
                    INSERT INTO predictions_daily
                        (asset_id, forecast_date, predicted_class,
                         p_low, p_medium, p_high,
                         bundle_version, data_cutoff_date, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        r["asset_id"], r["forecast_date"], r["predicted_class"],
                        r["p_low"], r["p_medium"], r["p_high"],
                        r["bundle_version"], r["data_cutoff_date"],
                        r["status"], r["updated_at"],
                    ],
                )
            total += len(rows)
        con.commit()
        return total
    finally:
        con.close()

def wipe(db_path: str) -> int:
    """Delete all rows marked bundle_version='synthetic-seed'. Returns deleted count."""
    con = duckdb.connect(db_path)
    try:
        con.execute(_DDL)  # ensure table exists
        result = con.execute(
            "SELECT COUNT(*) FROM predictions_daily WHERE bundle_version = ?", [MARKER]
        ).fetchone()
        count = result[0] if result else 0
        con.execute("DELETE FROM predictions_daily WHERE bundle_version = ?", [MARKER])
        con.commit()
        return count
    finally:
        con.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed or wipe synthetic predictions.")
    p.add_argument("--wipe", action="store_true", help="Delete synthetic rows instead of inserting.")
    p.add_argument("--db", default=DEFAULT_DB, help="Path to serving.duckdb.")
    p.add_argument("--days", type=int, default=90, help="Days of history to generate (default 90).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.wipe:
        n = wipe(args.db)
        print(f"Wiped {n} synthetic rows from {args.db}")
    else:
        n = seed(args.db, args.days)
        print(f"Inserted {n} synthetic rows into {args.db}")
