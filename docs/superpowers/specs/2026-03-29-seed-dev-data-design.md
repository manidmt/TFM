# Design: seed_dev_data.py — Synthetic Prediction Seed Script

**Date:** 2026-03-29
**Status:** Approved

## Purpose

A standalone development utility to populate `predictions_daily` in `serving.duckdb` with realistic synthetic data, enabling frontend development and UI design without the ML pipeline running. All inserted rows are marked with `bundle_version = 'synthetic-seed'` for trivial identification and removal.

## CLI Interface

```bash
# Insert synthetic data (idempotent — safe to re-run)
python scripts/seed_dev_data.py

# Wipe all synthetic rows
python scripts/seed_dev_data.py --wipe

# Point to a custom DB path
python scripts/seed_dev_data.py --db /path/to/serving.duckdb
```

## Assets

All 6 assets from `config/prod/assets.yaml`:

| asset_id | Display |
|---|---|
| `us_equities` | US Equities (S&P 500) |
| `euro_equities` | Euro Area Equities |
| `bitcoin` | Bitcoin |
| `long_us_treasuries` | Long US Treasuries |
| `short_us_treasuries` | Short US Treasuries |
| `gold` | Gold |

## Data Generation

**Volume:** 90 days of history per asset (today-90 to today), giving ~91 rows per asset (551 total).

**Regime simulation — Markov chain per asset:**

Base transition matrix (row = current, col = next: low / medium / high):

| | low | medium | high |
|---|---|---|---|
| low | 0.70 | 0.30 | 0.00 |
| medium | 0.20 | 0.60 | 0.20 |
| high | 0.00 | 0.40 | 0.60 |

Bitcoin override: higher baseline volatility (starts in `high`, elevated `high→high` = 0.70).

**Probabilities:** Sampled from `Dirichlet(α)` where α is regime-dependent:
- `low` regime: α = [6, 2, 0.5] → dominant low prob
- `medium` regime: α = [1.5, 5, 1.5] → dominant medium prob
- `high` regime: α = [0.5, 2, 6] → dominant high prob

This produces realistic-looking probability bars (non-uniform, plausible spreads).

**Seed:** Each asset uses a fixed seed derived from its name so results are reproducible across runs.

## Marker and Wipe

All rows inserted with `bundle_version = 'synthetic-seed'`. Wipe is a single SQL statement:

```sql
DELETE FROM predictions_daily WHERE bundle_version = 'synthetic-seed';
```

The script also initialises the schema (`CREATE TABLE IF NOT EXISTS`) so it works against a blank DB.

## Non-goals

- Does not touch `active_bundles`, `asset_status`, `prediction_runs`, or `promotion_events`.
- Does not import anything from `quant_risk` — only `duckdb` and stdlib.
- Not intended for production use; the marker makes accidental deployment detectable.

## Overwrite Behaviour

`predictions_daily` has PRIMARY KEY `(asset_id, forecast_date)`. Re-running the seed replaces existing synthetic rows. If real predictions exist for a date, they are also overwritten — this is acceptable in a dev environment. The `--wipe` flag restores the pre-seed state for any synthetic rows.
