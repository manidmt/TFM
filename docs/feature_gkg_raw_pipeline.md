# GKG Raw Pipeline (Canonical News Path)

For the frozen baseline used in tracked experiments, see
[gkg_v1.md](/home/manidmt/TFM/quant-risk-tfm/docs/gkg_v1.md). This document describes the
canonical pipeline architecture; `gkg_v1.md` defines the stable operational profile.

## What Changed

- Added a **new canonical news ingestion pipeline** based on GDELT **GKG raw**.
- Added new module: [`src/quant_risk/data/gkg.py`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/data/gkg.py)
- Added new script: [`scripts/ingest_gkg.py`](/home/manidmt/TFM/quant-risk-tfm/scripts/ingest_gkg.py)
- Integrated new daily news table into feature builder:
  - [`src/quant_risk/features/build.py`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/features/build.py)
  - [`scripts/build_features.py`](/home/manidmt/TFM/quant-risk-tfm/scripts/build_features.py)
- Kept DOC timeline path for backward compatibility, but marked as deprecated/exploratory:
  - [`scripts/ingest_gdelt.py`](/home/manidmt/TFM/quant-risk-tfm/scripts/ingest_gdelt.py)

## New Tables

### Raw table (canonical)
- `gdelt_gkg_raw`
- Core columns:
  - `gkg_id`
  - `date`, `datetime`
  - `source`, `url`, `language`
  - `tone`, `themes`
  - `locations`, `persons`, `organizations`
  - `query_id`
  - `source_file`, `inserted_at`

### Daily table (canonical)
- `news_features_daily`
- Columns:
  - `date`, `query_id`
  - `news_count`
  - `tone_mean`, `tone_std`, `tone_neg_share`
  - `source`, `inserted_at`

## New Config Blocks

### datasources
- [`config/datasources.yaml`](/home/manidmt/TFM/quant-risk-tfm/config/datasources.yaml)
- Added `gkg` block with:
  - `enabled`, `raw_table`, `daily_table`
  - `start`, `lookback_buffer_days`, `publication_lag_bdays`
  - retry/backoff controls
  - `topics` map (`query_id -> query expression`)

### features
- [`config/features.yaml`](/home/manidmt/TFM/quant-risk-tfm/config/features.yaml)
- `news_features` now supports:
  - `source_table`
  - `topics_enabled`
  - existing rolling/tone/shock toggles

## How to Run

### 1) Ingest canonical GKG raw

```bash
PYTHONPATH=src poetry run python scripts/ingest_gkg.py \
  --config config/datasources.yaml \
  --start 2020-01-01 \
  --end 2026-03-10
```

Optional filters:

```bash
PYTHONPATH=src poetry run python scripts/ingest_gkg.py \
  --config config/datasources.yaml \
  --topic_ids macro_us,fed_inflation
```

### 2) Build model features + labels

```bash
PYTHONPATH=src poetry run python scripts/build_features.py \
  --config_features config/features.yaml \
  --config_sources config/datasources.yaml
```

## Anti-Leakage Guarantees

- Publication lag is applied **before** rolling operations in the news block.
- Rolling features are computed only from as-of historical values.
- Feature builder uses date-aligned business-day index and merges lagged news series.
- No `vol_fwd` leakage introduced in exogenous/news pathways.

Code references:
- News load/alignment/lag/rolling: [`build.py`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/features/build.py)
- Canonical raw->daily rebuild: [`gkg.py`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/data/gkg.py)

## Added Tests

- [`tests/test_gkg_ingest_smoke.py`](/home/manidmt/TFM/quant-risk-tfm/tests/test_gkg_ingest_smoke.py)
- [`tests/test_gkg_daily_features.py`](/home/manidmt/TFM/quant-risk-tfm/tests/test_gkg_daily_features.py)
- [`tests/test_gkg_no_leakage_alignment.py`](/home/manidmt/TFM/quant-risk-tfm/tests/test_gkg_no_leakage_alignment.py)
- [`tests/test_build_features_with_gkg.py`](/home/manidmt/TFM/quant-risk-tfm/tests/test_build_features_with_gkg.py)

Validation status:
- `poetry run pytest -q` -> **84 passed, 1 skipped**.
