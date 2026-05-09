# Methodology Page — Design Spec

## Overview

A static, public-facing page (`/methodology`) that explains how the platform works — from data ingestion through model training to production serving. Aimed at two audiences: a curious visitor wanting a high-level understanding, and a thesis committee evaluating the academic rigor.

## Access

- Route: `/methodology` (public, no auth required)
- Not in the nav header. Discoverable via a link on the Landing page (hero or subtitle area).
- Landing link text: "Learn about the methodology →" or similar.

## Layout

- Long-scroll article, single column, `.container` (max-width 1120px)
- Sections separated by `border-top: 1px solid var(--line)` with vertical padding
- Typography: Newsreader for section titles, IBM Plex Sans for body, IBM Plex Mono for metrics/data
- Fully static — no API calls, no state, no interactivity
- Responsive: cards collapse to single column below 640px

## Sections

### 1. Overview (hero)

Smaller hero than Landing (no full-viewport). Title "Methodology" in Newsreader, followed by 2-3 sentences:

> This platform predicts 5-day forward volatility regimes for six financial assets using a chained econometric-to-machine-learning pipeline. It combines classical time-series models, macroeconomic indicators, and news signals to classify upcoming volatility as low, medium, or high.

### 2. Asset Universe

Grid of 6 cards (3×2 desktop, 1 column mobile). Each card:
- **Ticker** (e.g., `^GSPC`) — IBM Plex Mono, small
- **Name** (e.g., "US Equities") — IBM Plex Sans, bold
- **Asset class label** (e.g., "Equities", "Crypto", "Fixed Income", "Commodities") — small text, muted
- **Rationale** — one sentence on why included (e.g., "Broad US equity market benchmark")

Assets:
| Ticker | Name | Class | Rationale |
|--------|------|-------|-----------|
| ^GSPC | US Equities | Equities | Broad US equity market benchmark (S&P 500) |
| ^STOXX50E | Euro Area Equities | Equities | Major European equity exposure (EURO STOXX 50) |
| BTC-USD | Bitcoin | Crypto | High-volatility digital asset with distinct regime dynamics |
| GLD | Gold | Commodities | Traditional safe-haven and inflation hedge |
| TLT | Long US Treasuries | Fixed Income | Duration-sensitive government bond proxy |
| SHY | Short US Treasuries | Fixed Income | Low-duration safe asset, near-cash benchmark |

### 3. Pipeline

Horizontal flow diagram (CSS flexbox, not an image). Each step is a rounded box with a short label and 1-2 lines of description below:

```
[Raw Data] → [Feature Engineering] → [Econometric Chain] → [Tabular ML] → [Calibration & Gating] → [Prediction]
```

Steps:
1. **Raw Data** — Daily prices, macro indicators, news signals
2. **Feature Engineering** — Realised volatility, rolling stats, shock detection, news tone aggregation
3. **Econometric Chain** — SARIMAX conditional mean → GARCH/EGARCH/GJR-GARCH/HAR-RV conditional variance
4. **Tabular ML** — XGBoost or TabPFN classifier on merged feature set
5. **Calibration & Gating** — Isotonic calibration, persistence gating, confidence thresholds
6. **Prediction** — Calibrated probabilities for low/medium/high regime

On mobile (< 640px): vertical layout with downward arrows.

Connecting arrows: thin lines with `→` or CSS borders. No SVG library — pure CSS.

### 4. Models

Four subsections, each with a heading and 2-4 sentences of explanation.

#### 4a. Econometric Layer

First stage of the pipeline. For each asset, multiple chained configurations are evaluated:
- **SARIMAX** — conditional mean model (orders p,d,q, optional macro exogenous variables like net liquidity)
- **Volatility models** — four variants: GARCH, EGARCH (exponential), GJR-GARCH (threshold/asymmetric), and HAR-RV (heterogeneous autoregressive realised volatility). Each fitted with different distributional assumptions (Student's t, skewed-t) and aggregation methods
- The "promising" profile evaluates 8 structural configurations per asset; the "full" profile evaluates 32

#### 4b. Tabular ML Layer

Econometric features (conditional mean, variance, standardised residuals) are merged with macro and news features and fed into a supervised classifier:
- **XGBoost** — gradient boosting, robust and well-understood baseline
- **TabPFN** — prior-data fitted network, a neural few-shot learner that generalises well with limited training windows. More novel; frequently selected by the sweep runner

#### 4c. Sweep Runner & Model Selection

The system does not hand-pick a model. A systematic sweep explores the full combination space:
- **Optuna** TPE sampler with independent studies per asset, tracked in **MLflow**
- Search space: econometric variant × tabular model × hyperparameters → **100+ combinations** per asset
- Each trial runs a complete walk-forward evaluation across multiple temporal folds
- Optimisation target: **robust score** — mean delta macro F1 vs. the persistence baseline, penalised for instability across folds and rewarded for correctly predicting regime transitions

#### 4d. Blending, Calibration & Gating

The raw prediction passes through three post-processing layers:
- **Blending** — weighted combination (alpha) of the econometric chain and tabular model predictions, with optional confidence-adaptive weighting (beta)
- **Isotonic calibration** — maps raw model probabilities to well-calibrated probabilities
- **Persistence gating** — when model confidence is low (max probability below threshold), the system falls back to the persistence baseline. Optionally modulated by a **GKG change detector** — a binary classifier that predicts regime transitions using news signals, attenuating the blend when no change is expected

### 5. Data Sources

Three cards in a row (1 column mobile):

**Market Prices** — yfinance
Daily close prices for all six assets. Realised volatility computed over rolling windows (5, 10, 21 days). Source of the core target variable.

**Macroeconomic Indicators** — FRED (Federal Reserve Economic Data)
VIX, yield curve spreads, interest rates, net liquidity. Provides broader market context that price data alone cannot capture.

**News Signals** — GDELT Global Knowledge Graph
Global event tone, thematic volume, information shock detection. An alternative signal source independent of price action. Aggregated daily with publication-date lag handling.

### 6. Regime Classification

Explanation of the three-regime quantile binning:
- Realised volatility values over the training window are sorted and split at the 33rd and 66th percentiles
- Three resulting classes: **Low** (bottom third), **Medium** (middle third), **High** (top third)

Visual element: three horizontal bars in the platform's regime colors (`--low` green, `--medium` amber, `--high` red) showing the percentile ranges (0–33%, 33–66%, 66–100%). Labels below each bar.

One sentence noting that thresholds are recalculated per walk-forward fold to avoid lookahead bias.

### 7. Evaluation

Core principle: a model must outperform the persistence baseline (always predict the most recent observed regime).

Key metrics, presented as a compact list:
- **Delta macro F1 vs. persistence** — primary metric. How much better is the model than simply repeating the last regime?
- **High-vol recall** — does the model catch high-volatility regimes? Missing these is costly for risk management
- **Robust score** — cross-fold aggregate penalised for instability and rewarded for correct transition predictions
- **Walk-forward validation** — train/evaluate on advancing temporal windows to simulate real deployment with no data leakage

### 8. Infrastructure

Diagram of two nodes with a connecting arrow:

```
┌─────────────────────┐          push predictions         ┌─────────────────────┐
│     Workstation      │ ───────────────────────────────▸  │   Raspberry Pi 5    │
│                      │    POST /api/internal/predictions │                      │
│  Sweep runner        │                                   │  FastAPI + Postgres  │
│  Model training      │                                   │  Nginx + Cloudflare  │
│  MLflow tracking     │                                   │  DuckDB (serving)    │
│  Heavy compute       │                                   │  24/7 low-power      │
└─────────────────────┘                                   └─────────────────────┘
```

CSS-rendered, not an image. Two card-like boxes with an arrow between them.

Explanation:
- **Workstation** — runs sweeps, trains models, evaluates walk-forward folds. Promotes the best model configuration weekly.
- **Raspberry Pi 5** — serves the web application and API around the clock. Does not run inference — it only serves pre-computed predictions from DuckDB.
- **Sync** — the workstation pushes daily predictions to the RPi5 via an authenticated internal endpoint. A systemd timer on the RPi5 runs a daily worker for data ingestion (prices, macro, news).

### 9. Limitations & Disclaimer

Short paragraph:
- This is an academic research tool, not financial advice
- Predictions are probabilistic — they express tendency, not certainty
- Past performance does not guarantee future results
- Models trained on data from 2015 onwards — may not capture unprecedented regime dynamics

Visually: muted text, slightly smaller font, inside a subtle bordered box.

## Files

| File | Action |
|------|--------|
| `apps/web/src/pages/Methodology.tsx` | Create — page component |
| `apps/web/src/pages/Methodology.css` | Create — page styles |
| `apps/web/src/App.tsx` | Modify — add `/methodology` route |
| `apps/web/src/pages/Landing.tsx` | Modify — add link to methodology |

## No backend changes

This is a frontend-only feature. All content is static and hardcoded in the component.
