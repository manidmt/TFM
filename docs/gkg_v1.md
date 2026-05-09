# GKG v1 Stable Spec

## Purpose

`GKG v1` is the stable external-news baseline that should be treated as the canonical
news dataset for serious tracked experiments before moving to `MLflow`.

This freezes the data/profile contract, not the research space. In particular:

- the **data ingestion profile** is now stable
- the **daily news feature table** is now stable
- the **compact feature-engineering profile** is now stable
- sidecar model research (`logit` vs `xgb`, gating vs alpha, per-asset tuning) remains open

## Official Ingest Profile

Source of truth:
- [datasources.yaml](/home/manidmt/TFM/quant-risk-tfm/config/datasources.yaml)
- [gkg.py](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/data/gkg.py)
- [ingest_gkg.py](/home/manidmt/TFM/quant-risk-tfm/scripts/ingest_gkg.py)

Frozen profile:

- `profile_name = gkg_v1_light`
- `profile_version = 2026-03-13`
- `profile_mode = light_daily`
- `start = 2020-01-01`
- `store_raw = false`
- `sample_interval_minutes = 180`
- `max_files_per_day = 8`
- `max_files_total = 0`
- `publication_lag_bdays = 1`

Interpretation:

- `light_daily` means the canonical baseline stores the **daily aggregated table** and does
  not require full historical raw storage in DuckDB.
- `2020-01-01` is the official historical start for the stable baseline because it is the
  sustainable range for this repository's disk/runtime constraints.
- `max_files_total = 0` is intentional. The stable baseline should not silently truncate the
  dataset globally; only the per-day light sampling profile is fixed.

## Official Topic Set

The stable topic set is:

- `macro_us`
- `fed_inflation`
- `rates_yields`
- `crypto_market`
- `geopolitical_risk`

These are intentionally broad cross-asset baseline channels. They are stable enough for
tracked experiments, but they are not claimed to be optimal per asset.

## Official Feature Profile

Source of truth:
- [features.yaml](/home/manidmt/TFM/quant-risk-tfm/config/features.yaml)
- [build.py](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/features/build.py)

Frozen feature profile:

- `profile_name = gkg_v1_features_compact`
- `profile_version = 2026-03-13`
- `source_table = news_features_daily`
- `windows = [3, 10, 20]`
- `compact_mode = true`
- `include_interactions = true`
- `topic_share_enabled = true`
- `include_roll_sum = false`
- `include_roll_mean = false`
- `include_roll_std = false`
- `include_tone_std = false`
- `include_tone_neg_share = false`

Health policy:

- dead topics are dropped automatically
- the build fails if all topics are dead

## Manifest / Reproducibility

Each `scripts/ingest_gkg.py` run now records a manifest row in DuckDB:

- table: `gkg_ingest_runs`

It stores:

- profile name/version/mode
- date range
- topic ids and topic hash
- storage mode
- light sampling controls
- inserted row counts
- error count

This is the minimum reproducibility contract required before tracked `MLflow` experiments.

## What Is Stable vs What Is Still Research

Stable in `v1`:

- ingest profile
- topic set
- daily table schema
- compact feature block
- anti-leakage lag discipline

Still research:

- sidecar backend: `logit` vs `xgb`
- sidecar calibration choice
- gating vs alpha vs combined use
- per-asset alpha/gate search spaces
- per-asset topic refinement
- broader history or denser sampling profiles

## Operational Rule

If any of these change, this is **not** the same dataset anymore and the profile should be
version-bumped before comparing tracked runs as if they were equivalent:

- `start`
- `store_raw`
- `sample_interval_minutes`
- `max_files_per_day`
- topic definitions
- daily aggregation schema
- compact feature profile

## Recommended Baseline Commands

Ingest `GKG v1`:

```bash
PYTHONPATH=src poetry run python scripts/ingest_gkg.py \
  --config config/datasources.yaml \
  --start 2020-01-01 \
  --end 2026-03-10
```

Build features:

```bash
PYTHONPATH=src poetry run python scripts/build_features.py \
  --config_features config/features.yaml \
  --config_sources config/datasources.yaml
```

## Why This Is Enough To Move To MLflow

What was missing before `MLflow` was not more model tuning. It was the absence of a fixed
news-data contract. `GKG v1` closes that gap:

- the data profile is explicit
- the feature profile is explicit
- the ingest manifest is persisted
- the remaining open questions are now clearly research questions, not data-contract questions
