# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**quant-risk-tfm** is a quantitative finance research framework (Master's thesis) for predicting volatility regimes in financial assets (S&P 500, Bitcoin, US Treasuries) using chained econometric-to-tabular ML models, news signals (GDELT GKG), and macro features. Forecasting horizons: 5-day and 20-day.

## Environment & Commands

```bash
# Setup
poetry install
poetry shell

# Tests
poetry run pytest -q                                              # full suite
poetry run pytest tests/test_build_features_with_gkg.py -v       # single file
poetry run pytest --cov=quant_risk tests/                         # with coverage

# Formatting
poetry run black src/ scripts/ tests/

# Notebooks
poetry run jupyter lab notebooks/
```

## Data Pipeline (run sequentially)

```bash
poetry run python scripts/ingest_prices.py
poetry run python scripts/ingest_macro.py
poetry run python scripts/ingest_gkg.py --config config/datasources.yaml --start 2020-01-01
poetry run python scripts/build_features.py --config_features config/features.yaml --config_sources config/datasources.yaml
```

## Main Training Workflow

```bash
# Primary: walk-forward chain evaluation
poetry run python scripts/walk_forward_chain_tab.py \
  --tickers GSPC BTC-USD TLT \
  --horizon 5 20 \
  --tabular_model xgb \
  --grid_profile promising \
  --use_mlflow \
  --mlflow_experiment walk_forward_chain_tab
```

## Architecture

### Data Flow

```
Raw Sources (yfinance, FRED, GDELT)
    → DuckDB tables (raw_prices, macro_features, gdelt_gkg_raw)
    → build_features.py / build_labels.py
    → DuckDB (features_daily, labels_regime)
    → make_dataset.py [DatasetConfig]
    → Econometric chain (SARIMAX/GARCH) + tabular feature merge
    → Tabular model (XGBoost/TabPFN) per walk-forward fold
    → Calibration (Isotonic) + threshold tuning + persistence gating
    → Evaluation (macro_f1, delta vs persistence, high_vol_recall)
    → Artifacts in runs/ + MLflow logging
```

### Key Modules

- **`src/quant_risk/data/`** — Ingestion: `fetcher.py` (prices), `macro.py` (FRED), `gkg.py` (GDELT GKG with lag handling)
- **`src/quant_risk/features/`** — `build.py` (realized vol, macro transforms, shock detection, rolling news), `labels.py` (quantile-based regime labeling)
- **`src/quant_risk/datasets/make_dataset.py`** — Orchestrates econometric feature extraction and time-series splits
- **`src/quant_risk/models/econometric/`** — SARIMAX, GARCH/EGARCH/GJR-GARCH, HAR-RV
- **`src/quant_risk/models/tabular/`** — XGBoost, TabPFN, GKG change detector, TabNet, FT-Transformer
- **`src/quant_risk/models/baseline.py`** — Persistence predictor, Logistic Regression, Random Forest
- **`src/quant_risk/models/metrics.py`** — Evaluation: macro_f1, delta metrics vs persistence, transition-aware metrics

### Configuration

- **`config/datasources.yaml`** — Tickers, FRED series, GKG topic profiles, DuckDB path (`data/db/financial_data.duckdb`)
- **`config/features.yaml`** — Feature windows, target horizons, train/val/test split dates (train end: 2020-12-31, val end: 2023-12-31), official feature profile `gkg_v1_features_compact`
- **`config/grid_*.yaml`** — Hyperparameter grids for model search

### Walk-Forward Logic

Per fold in `scripts/walk_forward_chain_tab.py`:
1. Fit econometric chain (SARIMAX → GARCH/HAR)
2. Extract conditional mean/variance features
3. Train tabular model on merged feature set
4. Evaluate: robust score = delta macro F1 vs persistence (threshold > -0.01 for merge)
5. Final: gating (confidence-based switching), calibration, test-set evaluation

## Storage

- **DuckDB**: `data/db/financial_data.duckdb` — primary data store
- **Runs**: `runs/` — per-experiment outputs (CSV, JSON, YAML); 120+ historical runs
- **Models**: `models/` — trained artifacts
- **MLflow**: integration in progress on `feature/mlflow` branch (see `docs/mlflow_v1_plan.md`)

## Development Notes

- Python 3.10–3.12 managed via Poetry
- No CI/CD; manual test before merge: `pytest -q` + smoke test `walk_forward_chain_tab.py` on all three tickers
- All changes via PR; squash merge preferred for clean history
- 3-regime quantile binning (low/medium/high volatility) as classification targets
- `feature/mlflow` is the active development branch; `main` is protected
