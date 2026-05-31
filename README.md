# quant-risk-tfm

Master's thesis — AI & Analytics, UCM / MIOTI, 2026  
**Volatility Regime Prediction for Financial Assets using Econometric-ML Chains**

Predicts daily volatility regimes (Low / Medium / High) for 6 financial assets at 5-day and 20-day horizons, combining econometric models, macro features, and news signals (GDELT GKG). The full pipeline runs in production on a Raspberry Pi 5 with a React web application.

**Live app**: https://risk.manidmt.es

---

## Assets

| Asset | Ticker | Class | Test Macro F1 |
|---|---|---|---|
| S&P 500 | ^GSPC | Equity | 0.427 |
| Gold | GLD | Commodity | 0.387 |
| T-Bills | SHY | Short-term Fixed Income | 0.378 |
| Euro Stoxx 50 | ^STOXX50E | Equity | 0.365 |
| Bitcoin | BTC-USD | Crypto | 0.336 |
| T-Bonds | TLT | Long-term Fixed Income | 0.331 |

Random baseline (3 balanced classes): 0.333. Models add significant value on regime **transition days** (Δ Macro F1 +0.25 to +0.31 vs persistence baseline).

---

## Architecture

```
Raw Sources (yfinance, FRED, GDELT GKG)
    → DuckDB  (raw_prices, macro_features, gdelt_gkg_raw)
    → build_features.py / build_labels.py
    → DuckDB  (features_daily, labels_regime)
    → make_dataset.py  [DatasetConfig]
    → Econometric chain  (SARIMAX → GARCH/EGARCH/GJR-GARCH/HAR-RV)
    → Tabular model  (XGBoost / TabPFN) — walk-forward expanding window
    → Platt calibration + confidence gating (θ-sweep)
    → Evaluation  (Macro F1, Δ vs persistence, transition recall)
    → Bundle  → push to RPi5 → FastAPI + React app
```

### Key modules

| Path | Description |
|---|---|
| `src/quant_risk/data/` | Ingestion: prices (yfinance), macro (FRED), GKG (GDELT) |
| `src/quant_risk/features/` | Realized vol, macro transforms, shock detection, rolling news |
| `src/quant_risk/datasets/` | `make_dataset.py` — econometric feature extraction + time-series splits |
| `src/quant_risk/models/econometric/` | SARIMAX, GARCH/EGARCH/GJR-GARCH, HAR-RV |
| `src/quant_risk/models/tabular/` | XGBoost, TabPFN, TabNet, FT-Transformer, GKG change detector |
| `src/quant_risk/models/baseline.py` | Persistence predictor, Logistic Regression, Random Forest |
| `src/quant_risk/models/metrics.py` | Macro F1, delta vs persistence, transition-aware metrics |
| `src/quant_risk/prod/` | FastAPI app, bundle registry, portfolio analysis, risk adjustment |
| `scripts/walk_forward_chain_tab.py` | Main training + evaluation loop |
| `scripts/push_predictions_to_rpi5.py` | Daily push of predictions and labels to production |
| `apps/web/` | React + Vite frontend |

---

## Setup

```bash
poetry install
poetry shell
```

Python 3.10–3.12 managed via Poetry.

---

## Data Pipeline

Run sequentially:

```bash
poetry run python scripts/ingest_prices.py
poetry run python scripts/ingest_macro.py
poetry run python scripts/ingest_gkg.py --config config/datasources.yaml --start 2020-01-01
poetry run python scripts/build_features.py \
  --config_features config/features.yaml \
  --config_sources config/datasources.yaml
```

Data sources:
- **Prices**: yfinance — daily OHLCV for all 6 assets (2020–present)
- **Macro**: FRED — Fed net liquidity, yield curve, VIX, credit spreads
- **News**: GDELT GKG — 5 thematic profiles (macro_us, fed_inflation, rates_yields, crypto_market, geopolitical_risk); 1 business day publication lag enforced

Primary store: `data/db/financial_data.duckdb`

---

## Training

Walk-forward expanding window with Bayesian hyperparameter search (Optuna TPE):

```bash
poetry run python scripts/walk_forward_chain_tab.py \
  --tickers GSPC BTC-USD TLT \
  --horizon 5 20 \
  --tabular_model xgb \
  --grid_profile promising \
  --use_mlflow \
  --mlflow_experiment walk_forward_chain_tab
```

Per fold:
1. Fit econometric chain (SARIMAX → GARCH/HAR-RV)
2. Extract conditional mean/variance features (6 features per asset)
3. Train tabular model on merged feature set
4. Evaluate: robust score = Δ Macro F1 vs persistence − stability penalty + transition bonus
5. Select best variant; apply Platt calibration + confidence gating

Splits: train end 2020-12-31, val end 2023-12-31, test 2024+.

---

## Evaluation

Primary metric: **Macro F1** — unweighted F1 across 3 regime classes. Penalises class-blind models equally regardless of regime imbalance.

Key metrics saved per run (`runs/<name>/final_vs_persistence.csv`):

| Metric | Description |
|---|---|
| `chain_test_macro_f1` | Model Macro F1 on test set |
| `persistence_test_macro_f1` | Persistence baseline Macro F1 |
| `delta_test_macro_f1_vs_persistence` | Difference (primary selection criterion) |
| `transition_macro_f1_test` | Model Macro F1 on regime-change days only |
| `delta_transition_macro_f1_vs_persistence_test` | Transition-day edge vs persistence |
| `gating_rate_test` | Fraction of days where model defers to persistence |

Experiments tracked in MLflow (`mlruns/`).

---

## Production

### Infrastructure

- **Workstation**: model training, sweeps, MLflow, daily inference
- **Raspberry Pi 5** (24/7): FastAPI + PostgreSQL + DuckDB + Nginx via Docker Compose

```
ops/docker/
├── compose.rpi5.yml     # API + worker + postgres + nginx
├── Dockerfile.api
├── Dockerfile.worker
└── nginx.conf
```

Cloudflare Tunnel exposes the app publicly (`ops/cloudflared/`).

### Daily cycle

1. Workstation fetches new prices/macro/GKG
2. Runs inference with production bundles (`config/prod/`)
3. `push_predictions_to_rpi5.py` pushes predictions + labels to RPi5 via SSH/SCP
4. RPi5 API serves updated predictions immediately

### Bundle promotion

Each asset has a production bundle (`incoming/bundles/<asset>/`) containing:
- Serialised econometric chain + tabular model
- Calibration parameters (Platt)
- Gating threshold θ
- `manifest.json` with val_macro_f1 and model metadata

---

## Web Application

React + Vite SPA served by Nginx. Source: `apps/web/src/`.

| Feature | Description |
|---|---|
| **Predictions dashboard** | Current regime forecast per asset with confidence and 30-day sparkline |
| **Asset history** | Price chart with regime overlay and historical predictions |
| **Portfolio analysis** | Historical VaR 95%/99% conditioned on current regime; idiosyncratic risk adjustment |
| **Chat agent** | Streaming AI assistant with access to live predictions and portfolio data |
| **Admin / Ops** | Bundle management, user administration, system health |

---

## Tests

```bash
poetry run pytest -q                                         # full suite
poetry run pytest tests/test_build_features_with_gkg.py -v  # single file
poetry run pytest --cov=quant_risk tests/                    # with coverage
```

Test coverage spans: ingestion, feature building, label construction, econometric models, tabular models, calibration/gating, walk-forward smoke, production API, bundle registry, portfolio analysis, auth.

---

## Configuration

| File | Purpose |
|---|---|
| `config/datasources.yaml` | Tickers, FRED series, GKG topic profiles, DuckDB path |
| `config/features.yaml` | Feature windows, target horizons, split dates, feature profile |
| `config/grid_*.yaml` | Hyperparameter search grids (XGBoost, TabNet, FT-Transformer) |
| `config/prod/` | Production overrides for RPi5 paths and asset universe |

---

## Repository layout

```
src/quant_risk/     # core Python package
scripts/            # pipeline + training entrypoints
apps/web/           # React frontend
ops/                # Docker, systemd, Cloudflare config
config/             # YAML configuration
tests/              # pytest suite
notebooks/          # EDA
runs/               # walk-forward output artefacts (gitignored)
models/             # trained model artefacts (gitignored)
data/               # DuckDB + raw files (gitignored)
```

---

## Author

Manuel Díaz-Meco Terrés — manidmt5@gmail.com
