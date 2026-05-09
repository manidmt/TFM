# Seed Dev Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `scripts/seed_dev_data.py` — a standalone script that populates `predictions_daily` in `serving.duckdb` with 90 days of realistic synthetic predictions for 6 production assets, marked `bundle_version='synthetic-seed'` for easy removal.

**Architecture:** Single self-contained script. Generates per-asset regime sequences via a Markov chain, then samples class probabilities from a Dirichlet distribution conditioned on the active regime. All inserted rows share a sentinel `bundle_version` so a single `DELETE` removes them. No imports from `quant_risk`.

**Tech Stack:** Python 3.10+, `duckdb`, `argparse`, `random` (stdlib only)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `scripts/seed_dev_data.py` | CLI entry-point + data generation + DB writes |
| Create | `tests/prod/test_seed_dev_data.py` | Unit tests using in-memory DuckDB |

---

## Task 1: Scaffold the script and wire the CLI

**Files:**
- Create: `scripts/seed_dev_data.py`

- [ ] **Step 1: Create the script with assets list, CLI, and stub functions**

```python
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
from pathlib import Path

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
    raise NotImplementedError

def _dirichlet_sample(alpha: list[float], rng: random.Random) -> tuple[float, float, float]:
    raise NotImplementedError

def _generate_rows(asset_id: str, n_days: int) -> list[dict]:
    raise NotImplementedError

def seed(db_path: str, n_days: int = 90) -> int:
    raise NotImplementedError

def wipe(db_path: str) -> int:
    raise NotImplementedError


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
```

- [ ] **Step 2: Verify the script is importable (no syntax errors)**

```bash
cd /home/manidmt/TFM/quant-risk-tfm
python -c "import scripts.seed_dev_data" 2>&1 || python scripts/seed_dev_data.py --help
```

Expected: help text printed, no errors (stubs raise NotImplementedError but `--help` exits before calling them).

---

## Task 2: Implement core data-generation helpers

**Files:**
- Modify: `scripts/seed_dev_data.py`

- [ ] **Step 1: Write the failing tests for the three helpers**

Create `tests/prod/test_seed_dev_data.py`:

```python
"""Tests for scripts/seed_dev_data.py — uses in-memory DuckDB."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import random

import pytest

# ---------------------------------------------------------------------------
# Import the script as a module (it lives in scripts/, not a package)
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).parents[2] / "scripts" / "seed_dev_data.py"
spec = importlib.util.spec_from_file_location("seed_dev_data", _SCRIPT)
_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(_mod)  # type: ignore[union-attr]

markov_next = _mod._markov_next
dirichlet_sample = _mod._dirichlet_sample
generate_rows = _mod._generate_rows
REGIMES = _mod.REGIMES
BASE_TRANSITIONS = _mod.BASE_TRANSITIONS
DIRICHLET_ALPHA = _mod.DIRICHLET_ALPHA
MARKER = _mod.MARKER
ASSETS = _mod.ASSETS


# ---------------------------------------------------------------------------
# _markov_next
# ---------------------------------------------------------------------------

def test_markov_next_returns_valid_regime():
    rng = random.Random(42)
    result = markov_next("low", BASE_TRANSITIONS, rng)
    assert result in REGIMES


def test_markov_next_deterministic_with_same_seed():
    rng1 = random.Random(99)
    rng2 = random.Random(99)
    r1 = markov_next("medium", BASE_TRANSITIONS, rng1)
    r2 = markov_next("medium", BASE_TRANSITIONS, rng2)
    assert r1 == r2


def test_markov_next_zero_prob_transition_never_happens():
    # low→high has prob 0.0 in BASE_TRANSITIONS
    rng = random.Random(0)
    results = {markov_next("low", BASE_TRANSITIONS, rng) for _ in range(200)}
    assert "high" not in results


# ---------------------------------------------------------------------------
# _dirichlet_sample
# ---------------------------------------------------------------------------

def test_dirichlet_sample_sums_to_one():
    rng = random.Random(7)
    p_low, p_med, p_high = dirichlet_sample(DIRICHLET_ALPHA["low"], rng)
    assert abs(p_low + p_med + p_high - 1.0) < 1e-9


def test_dirichlet_sample_all_positive():
    rng = random.Random(7)
    for regime in REGIMES:
        p = dirichlet_sample(DIRICHLET_ALPHA[regime], rng)
        assert all(v > 0 for v in p)


def test_dirichlet_low_regime_dominant():
    rng = random.Random(42)
    # average over many samples — p_low should dominate
    total_low = sum(dirichlet_sample(DIRICHLET_ALPHA["low"], rng)[0] for _ in range(500))
    assert total_low / 500 > 0.5


def test_dirichlet_high_regime_dominant():
    rng = random.Random(42)
    total_high = sum(dirichlet_sample(DIRICHLET_ALPHA["high"], rng)[2] for _ in range(500))
    assert total_high / 500 > 0.5


# ---------------------------------------------------------------------------
# _generate_rows
# ---------------------------------------------------------------------------

def test_generate_rows_count():
    rows = generate_rows("us_equities", 30)
    assert len(rows) == 31   # 30 days back + today


def test_generate_rows_fields():
    rows = generate_rows("us_equities", 10)
    required = {"asset_id", "forecast_date", "predicted_class",
                "p_low", "p_medium", "p_high", "bundle_version",
                "data_cutoff_date", "status", "updated_at"}
    assert required.issubset(rows[0].keys())


def test_generate_rows_marker():
    rows = generate_rows("bitcoin", 5)
    assert all(r["bundle_version"] == MARKER for r in rows)


def test_generate_rows_probs_sum_to_one():
    rows = generate_rows("gold", 20)
    for r in rows:
        assert abs(r["p_low"] + r["p_medium"] + r["p_high"] - 1.0) < 1e-9


def test_generate_rows_reproducible():
    rows1 = generate_rows("us_equities", 10)
    rows2 = generate_rows("us_equities", 10)
    assert [r["predicted_class"] for r in rows1] == [r["predicted_class"] for r in rows2]


def test_generate_rows_dates_ascending():
    rows = generate_rows("gold", 10)
    dates = [r["forecast_date"] for r in rows]
    assert dates == sorted(dates)
```

- [ ] **Step 2: Run tests to confirm they all fail with NotImplementedError**

```bash
cd /home/manidmt/TFM/quant-risk-tfm
poetry run pytest tests/prod/test_seed_dev_data.py -v 2>&1 | head -40
```

Expected: all tests FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `_markov_next`**

Replace the `_markov_next` stub in `scripts/seed_dev_data.py`:

```python
def _markov_next(current: str, transitions: dict[str, list[float]], rng: random.Random) -> str:
    """Sample next regime using the transition row for `current`."""
    weights = transitions[current]
    return rng.choices(REGIMES, weights=weights, k=1)[0]
```

- [ ] **Step 4: Implement `_dirichlet_sample`**

Replace the `_dirichlet_sample` stub:

```python
def _dirichlet_sample(alpha: list[float], rng: random.Random) -> tuple[float, float, float]:
    """Sample a probability triple from Dirichlet(alpha) using the gamma trick."""
    gammas = [rng.gammavariate(a, 1.0) for a in alpha]
    total = sum(gammas)
    p = [g / total for g in gammas]
    return (p[0], p[1], p[2])
```

- [ ] **Step 5: Implement `_generate_rows`**

Replace the `_generate_rows` stub:

```python
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
            "p_low": round(p_low, 6),
            "p_medium": round(p_med, 6),
            "p_high": round(p_high, 6),
            "bundle_version": MARKER,
            "data_cutoff_date": forecast_date,
            "status": "success",
            "updated_at": now,
        })
        regime = _markov_next(regime, transitions, rng)

    return rows
```

- [ ] **Step 6: Run the helper tests — they should all pass**

```bash
cd /home/manidmt/TFM/quant-risk-tfm
poetry run pytest tests/prod/test_seed_dev_data.py -v -k "markov or dirichlet or generate"
```

Expected: 12 tests PASS.

---

## Task 3: Implement `seed` and `wipe` functions

**Files:**
- Modify: `scripts/seed_dev_data.py`
- Modify: `tests/prod/test_seed_dev_data.py`

- [ ] **Step 1: Add DB-level tests to the test file**

Append to `tests/prod/test_seed_dev_data.py`:

```python
# ---------------------------------------------------------------------------
# seed / wipe (in-memory DuckDB)
# ---------------------------------------------------------------------------

seed_fn = _mod.seed
wipe_fn = _mod.wipe
_DDL = _mod._DDL
import tempfile, os

def _make_tmp_db() -> str:
    """Return path to a fresh temp DuckDB file."""
    fd, path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    return path


def test_seed_inserts_rows():
    import duckdb
    path = _make_tmp_db()
    try:
        n = seed_fn(path, n_days=10)
        assert n == 11 * len(ASSETS)   # 11 rows per asset (10 days + today)
        con = duckdb.connect(path)
        count = con.execute("SELECT COUNT(*) FROM predictions_daily").fetchone()[0]
        con.close()
        assert count == n
    finally:
        os.unlink(path)


def test_seed_all_rows_marked():
    import duckdb
    path = _make_tmp_db()
    try:
        seed_fn(path, n_days=5)
        con = duckdb.connect(path)
        unmarked = con.execute(
            f"SELECT COUNT(*) FROM predictions_daily WHERE bundle_version != '{MARKER}'"
        ).fetchone()[0]
        con.close()
        assert unmarked == 0
    finally:
        os.unlink(path)


def test_seed_idempotent():
    import duckdb
    path = _make_tmp_db()
    try:
        n1 = seed_fn(path, n_days=5)
        n2 = seed_fn(path, n_days=5)
        con = duckdb.connect(path)
        count = con.execute("SELECT COUNT(*) FROM predictions_daily").fetchone()[0]
        con.close()
        assert count == n1 == n2
    finally:
        os.unlink(path)


def test_wipe_removes_synthetic_rows():
    import duckdb
    path = _make_tmp_db()
    try:
        seed_fn(path, n_days=5)
        wiped = wipe_fn(path)
        assert wiped == 6 * 6   # 6 assets × 6 rows each
        con = duckdb.connect(path)
        count = con.execute("SELECT COUNT(*) FROM predictions_daily").fetchone()[0]
        con.close()
        assert count == 0
    finally:
        os.unlink(path)


def test_wipe_on_empty_db_returns_zero():
    path = _make_tmp_db()
    try:
        import duckdb
        con = duckdb.connect(path)
        con.execute(_DDL)
        con.close()
        wiped = wipe_fn(path)
        assert wiped == 0
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run these new tests to confirm they fail**

```bash
cd /home/manidmt/TFM/quant-risk-tfm
poetry run pytest tests/prod/test_seed_dev_data.py -v -k "seed or wipe" 2>&1 | head -30
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `seed`**

Replace the `seed` stub in `scripts/seed_dev_data.py`:

```python
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
                    """
                    INSERT OR REPLACE INTO predictions_daily
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
```

- [ ] **Step 4: Implement `wipe`**

Replace the `wipe` stub:

```python
def wipe(db_path: str) -> int:
    """Delete all rows marked bundle_version='synthetic-seed'. Returns deleted count."""
    con = duckdb.connect(db_path)
    try:
        con.execute(_DDL)  # ensure table exists
        result = con.execute(
            f"SELECT COUNT(*) FROM predictions_daily WHERE bundle_version = '{MARKER}'"
        ).fetchone()
        count = result[0] if result else 0
        con.execute(f"DELETE FROM predictions_daily WHERE bundle_version = '{MARKER}'")
        con.commit()
        return count
    finally:
        con.close()
```

- [ ] **Step 5: Run all tests**

```bash
cd /home/manidmt/TFM/quant-risk-tfm
poetry run pytest tests/prod/test_seed_dev_data.py -v
```

Expected: all 21 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/seed_dev_data.py tests/prod/test_seed_dev_data.py
git commit -m "feat: add seed_dev_data script for synthetic prediction seeding"
```

---

## Task 4: Smoke-test against local DB

**Files:** none — runtime verification only

- [ ] **Step 1: Run the seed against the local serving.duckdb**

```bash
cd /home/manidmt/TFM/quant-risk-tfm
poetry run python scripts/seed_dev_data.py --db data/db/serving.duckdb
```

Expected output:
```
Inserted 546 synthetic rows into data/db/serving.duckdb
```
(6 assets × 91 rows = 546)

- [ ] **Step 2: Verify via DuckDB CLI that the API will return data**

```bash
poetry run python -c "
import duckdb
con = duckdb.connect('data/db/serving.duckdb')
print(con.execute(\"SELECT asset_id, COUNT(*) n FROM predictions_daily GROUP BY 1 ORDER BY 1\").fetchall())
print('Latest:', con.execute(\"SELECT asset_id, forecast_date, predicted_class FROM predictions_daily WHERE forecast_date = (SELECT MAX(forecast_date) FROM predictions_daily) ORDER BY 1\").fetchall())
"
```

Expected: 6 rows in the count query (91 each), and 6 rows in the latest query with today's date.

- [ ] **Step 3: Hit the API to confirm end-to-end**

```bash
curl -s http://localhost:8000/api/public/predictions/latest | python -m json.tool | head -40
```

Expected: JSON array with 6 prediction objects.

- [ ] **Step 4: Wipe and confirm clean**

```bash
poetry run python scripts/seed_dev_data.py --wipe --db data/db/serving.duckdb
poetry run python -c "
import duckdb
con = duckdb.connect('data/db/serving.duckdb')
print(con.execute(\"SELECT COUNT(*) FROM predictions_daily WHERE bundle_version='synthetic-seed'\").fetchone())
"
```

Expected: `(0,)`
