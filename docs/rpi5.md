# RPi5 Deployment Architecture for Volatility Forecasting Web App

**STATUS:** Reviewed and aligned for V1 after design interview  
**Date:** March 27, 2026

---

## 1. Executive Summary

This document defines the **V1 production architecture** for deploying a volatility forecasting web application on a **Raspberry Pi 5**.

The final design is intentionally split into two lanes:

- **Research lane (powerful local machine):** experimentation, MLflow, model comparison, training, champion selection, and weekly bundle packaging
- **Production lane (RPi5):** daily data refresh, production feature build, inference, serving, user authentication, portfolio storage, and operations visibility

The RPi5 is **active**, but only for production inference workflows:

- daily ingestion of market, macro, and GKG/news data
- daily rebuild of **production-only** features
- daily inference using the latest promoted bundle per asset
- web serving for public predictions and private portfolio analysis

The RPi5 does **not** perform training, model selection, or broad experiment comparison.

V1 scope is intentionally narrow:

- **6 production proxies**
- **5-day horizon only**
- **1 promoted champion bundle per asset**
- **public prediction dashboard + public historical predictions**
- **private authenticated portfolio analysis**
- **admin-only read-only ops view**

---

## 2. V1 Product Scope

### Public

- Landing page
- Latest volatility prediction per production asset
- Full historical prediction view per production asset
- High-level explanation of the project and methodology

### Authenticated User Area

- Login/logout
- Multiple saved portfolios per user
- Manual portfolio construction by positions and weights
- Proxy mapping from user instruments to production asset proxies
- Simple portfolio what-if analysis

### Admin Area

- Read-only `/ops` view
- Batch status by asset
- Promotion status by asset
- Active bundle by asset
- System health summaries

### Explicitly Out of Scope for V1

- Chat / LLM assistant
- Public API for third-party developers
- Open self-service registration
- OAuth / social login
- Email-based password recovery
- Redis
- Prometheus / Grafana
- Public model diagnostics UI
- Cross-asset production feature engineering
- 20-day production horizon
- Training or model comparison on the RPi5

---

## 3. Core Decisions

| Topic | Final V1 Decision |
|------|-------------------|
| Production node | RPi5 is active for ingestion, feature build, inference, and serving |
| Training | Only on powerful local machine |
| Horizon | 5-day only |
| Production assets | 6 proxies from production config |
| Model promotion | Weekly, asset-scoped, independent per asset |
| Runtime bundles | `current` + `previous` per asset |
| Daily execution | Single D-1 batch run via `systemd timer` |
| Failure mode | Partial success by asset |
| Production features | Per-asset, self-contained, no cross-asset dependence |
| Research vs production | Separate lanes, separate configs, separate bundle contract |
| Public/private split | Predictions public; portfolios private |
| Auth | Email/password, server-side sessions, controlled account creation |
| Roles | `admin`, `user` |
| Ops UI | Read-only `/ops`, admin only |
| Data stores | `financial_data.duckdb`, `serving.duckdb`, `Postgres` |
| Reverse proxy | Existing host `Nginx` |
| Frontend runtime | React + Vite static build served by host Nginx |
| Containers | `api`, `worker`, `postgres` |
| Web container | Not required as a permanent runtime service |
| Chat | Phase 2 |

---

## 4. Production Asset Universe

V1 production uses the six assets already defined in the production universe.

The system uses a stable **semantic `asset_id`**, not a raw ticker, as the product-level identifier.

| asset_id | Display Name | Source Ticker | Category |
|---------|--------------|---------------|----------|
| `us_equities` | US Equities | `^GSPC` | Equity |
| `euro_equities` | Euro Area Equities | `^STOXX50E` | Equity |
| `bitcoin` | Bitcoin | `BTC-USD` | Crypto |
| `long_us_treasuries` | Long US Treasuries | `TLT` | Rates |
| `short_us_treasuries` | Short US Treasuries | `SHY` | Rates |
| `gold` | Gold | `GLD` | Commodity |

Each asset should also have a small catalog entry in app config with:

- `asset_id`
- `display_name`
- `source_ticker`
- `category`
- `short_description`

This catalog is versioned in the repo, not stored in a database.

---

## 5. High-Level Architecture

### 5.1 Runtime Topology on RPi5

```
                         risk.manidmt.es
                                |
                        Host Nginx (edge)
                    - serves static frontend
                    - proxies /api to FastAPI
                                |
        ---------------------------------------------------
        |                         |                       |
   React static build        FastAPI container     Worker container
   (built from repo)         (app backend)         (batch + ops jobs)
                                                        |
                                                        |
                                      -----------------------------------
                                      |                |                |
                              financial_data.duckdb   serving.duckdb    Postgres
                              raw+features         predictions+ops    users+portfolios+sessions
                                                        |
                                                 bundles/<asset_id>/
                                                 current + previous
```

### 5.2 Weekly Release Flow

```
Powerful machine
  -> train/select champions in production lane
  -> package one release containing 6 independent bundles
  -> transfer release.tar.gz to RPi5 incoming/

RPi5
  -> validate each bundle on-device
  -> smoke inference with local features
  -> promote passing assets independently
  -> keep current + previous pointers
```

### 5.3 Daily Batch Flow

```
systemd timer
  -> launches worker container
  -> ingest data
  -> rebuild production features
  -> load active bundle per asset
  -> infer 5d probabilities/classes
  -> update serving.duckdb
  -> expose latest + historical predictions to app
```

---

## 6. Separation of Lanes

### Research Lane

Runs on the powerful local machine and remains free to evolve.

Includes:

- broad experimentation
- MLflow
- model comparison
- cross-asset feature experiments
- wider hyperparameter sweeps
- exploratory diagnostics

### Production Lane

Strict, stable, and deployable.

Includes:

- exactly 6 production assets
- exactly 5-day horizon
- production-only configs
- per-asset feature contract
- champion selection under production constraints
- bundle packaging for RPi5

Important consequence:

**Production champions must be trained and validated against the same production feature contract they will use on the RPi5.**

That means production bundles should not rely on cross-asset features or other research-only dependencies.

---

## 7. Production Feature Policy

Production feature engineering is intentionally **per-asset and self-contained**.

Each asset may use:

- its own price history
- macro features
- GKG/news features

Each asset may **not** depend on:

- another asset's latest availability
- cross-asset correlations/spreads
- research-only feature blocks

This choice was made to preserve:

- true partial success by asset
- independent weekly promotion by asset
- simpler production inference contracts

If a production champion uses GKG/news features and those are temporarily stale, the system may internally mark the inference as using stale news, but that operational status is not part of the public UI.

---

## 8. Bundle Contract

### 8.1 Bundle Shape

Each promoted model is not a loose `.pkl`, but a **versioned bundle directory**.

Recommended structure:

```text
bundles/
  us_equities/
    current -> 2026-03-27_prod_ab12cd3
    previous -> 2026-03-20_prod_f45e9aa
    2026-03-27_prod_ab12cd3/
      manifest.json
      model/
        ...
      feature_contract.json
      inference_config.json
      calibration.json
      thresholds.json
```

### 8.2 Bundle Manifest

Each bundle manifest should include at least:

- `release_id`
- `bundle_version`
- `asset_id`
- `source_ticker`
- `horizon_days`
- `model_type`
- `feature_profile`
- `training_run_ref`
- `created_at`
- `python_version`
- dependency versions
- checksum / integrity metadata

### 8.3 Release Packaging

Weekly transport is:

- one `release.tar.gz`
- containing 6 independent bundles, one per `asset_id`
- one shared `release_id`, for example `prod_2026-03-27`

### 8.4 Validation on RPi5

Promotion requires on-device validation:

- checksum / manifest integrity
- compatibility checks
- bundle load succeeds
- smoke inference with recent local production features
- latency / resource check within acceptable bounds

If validation fails for one asset:

- that asset is **not** promoted
- the currently active bundle remains in place
- the other assets may still promote successfully

---

## 9. Promotion Model

Promotion is **independent by asset**, not all-or-nothing.

For each `asset_id`:

- `current` points to the active bundle
- `previous` points to the last known-good bundle

Promotion logic:

1. unpack release into staging
2. validate one asset bundle
3. if validation passes:
   - move old `current` to `previous`
   - repoint `current` to new bundle
   - update `active_bundles` state in `serving.duckdb`
4. if validation fails:
   - do not change `current`
   - mark promotion failure in ops metadata

No HTTP promotion endpoint is needed in V1. Promotion is an internal worker/CLI operation.

---

## 10. Daily Batch Semantics

### 10.1 Scheduler

V1 uses:

- host-level `systemd timer`
- which launches the worker container/job

V1 does **not** use:

- APScheduler in FastAPI
- cron as primary mechanism

### 10.2 Batch Frequency

V1 runs:

- **one daily batch**
- based on **D-1 closed data**

No intraday support is assumed in V1.

### 10.3 Batch Responsibilities

The worker should:

1. ingest prices
2. ingest macro data
3. ingest GKG/news data
4. build production features
5. load active bundle per asset
6. run inference for 5-day horizon
7. write prediction outputs and run state
8. update operational status for `/ops`

### 10.4 Failure Policy

Daily execution is **partial success by asset**.

Possible internal statuses per asset:

- `fresh`
- `stale_news`
- `stale_data`
- `failed`

These statuses matter operationally, but the public UI should remain cleaner and avoid surfacing raw staleness flags directly.

### 10.5 Rerun Policy

For a given `asset_id + forecast_date`:

- there is one logical final prediction row
- reruns update/replace that final row
- attempts are logged separately in a run/attempt table

Recommended retry policy:

- one scheduled daily run
- at most one small automatic retry for clearly transient failures
- manual rerun for the rest

---

## 11. Data Stores

### 11.1 `financial_data.duckdb`

Purpose:

- raw market data
- macro data
- GKG/news data
- production feature store

Examples:

- `raw_prices`
- `macro_features`
- `news_features_daily`
- production feature tables

### 11.2 `serving.duckdb`

Purpose:

- active bundle state
- prediction history
- prediction run metadata
- asset-level serving state

Recommended logical tables:

- `active_bundles`
- `prediction_runs`
- `predictions_daily`
- `asset_status`
- `promotion_events`

### 11.3 `Postgres`

Purpose:

- users
- roles
- sessions
- portfolios
- positions

Recommended logical tables:

- `users`
- `sessions`
- `portfolios`
- `portfolio_positions`

### 11.4 Why Three Stores

- `DuckDB` fits analytical/local serving data well
- `Postgres` is the right home for users, sessions, and portfolios
- separating `financial_data` from `serving` reduces coupling between write-heavy batch work and read-heavy app traffic

---

## 12. Web App Structure

V1 uses a **single React application** with three route zones:

- `/` public
- `/app/*` authenticated user area
- `/ops/*` admin-only read-only area

This is one SPA, not three separate frontends.

### 12.1 Public Zone

Public pages should include:

- landing page
- latest prediction cards/views per asset
- historical prediction views per asset
- explanatory product copy

Public prediction history can show:

- `forecast_date`
- `predicted_class`
- class probabilities

Public pages should **not** show:

- `bundle_version`
- `model_type`
- stale/failure operational flags

### 12.2 Authenticated User Zone

Authenticated pages should include:

- login/logout
- portfolio list
- portfolio detail/edit view
- what-if analysis

### 12.3 Admin Zone

`/ops/*` is protected by `admin` role.

V1 `/ops` is read-only and may include:

- latest batch status by asset
- latest prediction timestamp by asset
- active bundle by asset
- recent promotion results
- system health summaries
- recent error/log summaries

It should **not** allow:

- bundle promotion
- rerun trigger
- user creation
- role edits
- destructive data changes

---

## 13. Backend and Runtime

### 13.1 FastAPI

FastAPI exists as the **internal backend for the web app**, not as a public developer API product.

One backend service is enough for V1, modularized internally by domain:

- public prediction reads
- auth/session handling
- portfolio logic
- admin/ops reads

### 13.2 Worker

The worker is a separate runtime process from the API:

- same repo
- may share image/base code
- different entrypoint
- no scheduler embedded into the API server

### 13.3 Postgres Runtime

Postgres runs as a container on the RPi5 with persistent volume storage.

### 13.4 Frontend Runtime

- React + Vite
- built to static files
- served by existing host Nginx
- no permanent `web` container required

### 13.5 Reverse Proxy

The existing host `Nginx` remains the single edge proxy.

No separate in-Docker reverse proxy is needed for V1.

Deployment target:

- `risk.manidmt.es`

---

## 14. Authentication and Authorization

### 14.1 Auth Model

V1 uses:

- `email + password`
- server-side sessions
- session cookies
- session persistence in Postgres

V1 does not use:

- Basic Auth as primary product auth
- OAuth
- magic links
- public sign-up

### 14.2 Account Lifecycle

- accounts are created manually by admin
- initial bootstrap admin is created by CLI
- further users/admins are created by internal CLI/admin command
- password reset is manual by admin
- new users receive temporary credentials and must change password at first login

### 14.3 Roles

V1 defines only:

- `user`
- `admin`

### 14.4 Sessions

Recommended V1 session policy:

- cookie-based session
- persisted in Postgres
- valid for about 14 days
- renewed on activity

---

## 15. Portfolio V1

### 15.1 Goal

Portfolio analysis in V1 is not a full multi-asset risk engine.

It is a **proxy-based aggregation layer** built on top of the six production prediction assets.

### 15.2 User Input

Portfolio input is manual and long-only:

- one row per user position
- freeform label/ticker, for example `AAPL` or `BBVA`
- weight in percent
- manual proxy assignment to one `asset_id`

Examples:

- `AAPL` -> `us_equities`
- `BBVA` -> `euro_equities`

### 15.3 Portfolio Rules

V1 portfolios are:

- long-only
- manually weighted
- no leverage
- no negative weights
- normalized to ~100%

### 15.4 Portfolio Persistence

- portfolios are stored server-side in Postgres
- each user may have multiple portfolios
- each portfolio stores only its **current state** in V1
- no detailed version history yet

### 15.5 Analysis Semantics

Backend portfolio analysis should:

1. validate and normalize weights
2. resolve proxy per row
3. aggregate weights by `asset_id`
4. join latest production prediction for each proxy
5. compute interpretable aggregate outputs
6. support simple what-if changes and presets

### 15.6 What V1 Should Not Pretend To Be

V1 should not market itself as:

- a full institutional risk system
- Monte Carlo engine
- precise volatility-annualization oracle

The main signal remains:

- predicted class (`low`, `medium`, `high`)
- class probabilities
- interpretable aggregate portfolio signal

---

## 16. Prediction Contract

V1 uses exactly three canonical classes for every production asset:

- `low`
- `medium`
- `high`

The primary user-facing output is:

- predicted class
- probability distribution across the three classes

The public and private product should center on these outputs rather than pretending to expose a single exact volatility number as the main truth.

---

## 17. Public vs Private Surface

### Public

- landing
- latest predictions
- full historical predictions

### Private

- login/logout
- user portfolio management
- what-if analysis

### Admin

- `/ops`

This split is important because it replaces the earlier idea of global Basic Auth.

---

## 18. Example Internal API Shape

This backend is internal to the site, but a route structure like this is recommended.

### Public

- `GET /api/public/assets`
- `GET /api/public/predictions/latest`
- `GET /api/public/predictions/history`

### Auth

- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/auth/me`
- `POST /api/auth/change-password`

### Private

- `GET /api/private/portfolios`
- `POST /api/private/portfolios`
- `PUT /api/private/portfolios/{id}`
- `DELETE /api/private/portfolios/{id}`
- `POST /api/private/portfolios/{id}/analyze`

### Admin

- `GET /api/admin/ops/summary`
- `GET /api/admin/ops/assets`
- `GET /api/admin/ops/promotions`

Notably absent in V1:

- public model promotion endpoint
- public bundle management endpoints
- chat endpoints

---

## 19. App Release vs Model Release

These are **separate** processes.

### App Release

- same repo cloned on the RPi5
- build happens on the RPi5
- deploy by checking out a chosen commit/tag and rebuilding runtime services

This is intentionally simple for V1 and avoids premature CI/CD complexity.

### Model Release

- produced on the powerful local machine
- transferred as `release.tar.gz`
- copied to `incoming/` on the RPi5
- validated and promoted asset by asset

App releases must not be tied to weekly model refreshes.

---

## 20. Operations and Observability

V1 observability is intentionally pragmatic.

### Included

- worker logs
- API logs
- health endpoints
- operational state in `serving.duckdb`
- read-only `/ops` view

### Not Included

- Prometheus
- Grafana
- Redis-backed queues
- heavy observability stack

This is sufficient for V1 because the most important thing is to know:

- what ran
- what failed
- which bundle is active
- which prediction is currently being served

---

## 21. Backup and Retention Policy

Recommended V1 policy:

- **daily backup** of `Postgres`
- **daily backup** of `serving.duckdb`
- **weekly backup** of `financial_data.duckdb`
- retain `current + previous` bundles on hot storage
- include both in backup
- backup destination should be external to the same SSD when possible

This is enough to cover the three most important recovery scenarios:

- account/portfolio loss
- serving-state corruption
- bundle rollback/recovery

---

## 22. Suggested Runtime Layout on RPi5

Example host layout:

```text
/opt/quant-risk-tfm/          # repo clone used to build/deploy app code
/srv/quant-risk/
  incoming/                   # weekly model releases land here
  bundles/                    # current + previous by asset
  db/
    financial_data.duckdb
    serving.duckdb
  postgres/
    ...
  static/                     # built frontend assets for Nginx
  logs/
    ...
```

---

## 23. Open Items for Phase 2

Not part of V1, but explicitly expected later:

- chat / agentic assistant
- richer admin tooling
- self-service registration or invitation flow
- automated password recovery
- deeper model diagnostics
- possibly broader public product surface
- more advanced portfolio analytics

---

## 24. Final V1 Summary

V1 is a **public/private web app** on `risk.manidmt.es` with:

- public volatility predictions and historical views
- private authenticated portfolios
- admin-only operational visibility

The **RPi5 is active**, but only for production operations:

- ingest data
- build production features
- run inference
- serve predictions

The **powerful local machine** remains responsible for:

- research
- training
- comparison
- champion selection
- weekly bundle packaging

The production runtime is intentionally small and coherent:

- host Nginx
- static React frontend
- FastAPI backend
- worker
- Postgres
- `financial_data.duckdb`
- `serving.duckdb`
- versioned asset bundles with `current + previous`

This is the architecture that the rest of the implementation should now target.

---

## 25. Implementation Checklist by Phase

### Phase 1: Production Foundations

- [ ] Create production-specific config files separate from research configs
- [ ] Formalize the six production `asset_id` entries and their catalog metadata
- [ ] Define the production feature contract for per-asset inference
- [ ] Define bundle manifest schema and bundle directory structure
- [ ] Define `release_id` and `bundle_version` naming convention
- [ ] Create runtime directory layout on the RPi5 under `/srv/quant-risk/`
- [ ] Prepare the RPi5 repo clone used for app builds and deploys
- [ ] Add/update `.dockerignore` and build contexts so app builds stay lean

### Phase 2: Data and Batch Runtime

- [ ] Reuse the existing `financial_data.duckdb` base tables (`raw_prices`, `macro_features`, `news_features_daily`) and add only missing production-side tables if needed
- [ ] Create `serving.duckdb` production schema/tables
- [ ] Implement worker entrypoint separate from research CLIs
- [ ] Reuse ingestion logic for prices in the production worker
- [ ] Reuse ingestion logic for macro data in the production worker
- [ ] Reuse ingestion logic for GKG/news in the production worker
- [ ] Implement production-only feature build path without cross-asset features
- [ ] Implement daily inference flow for 5-day horizon only
- [ ] Implement per-asset partial success handling and internal status codes
- [ ] Implement idempotent rerun behavior for `asset_id + forecast_date`
- [ ] Persist latest and historical predictions into `serving.duckdb`
- [ ] Persist batch runs, attempts, and promotion events into `serving.duckdb`

### Phase 3: Bundle Validation and Promotion

- [ ] Implement weekly release unpacking into `incoming/` and staging
- [ ] Implement per-bundle checksum and manifest validation
- [ ] Implement compatibility checks for promoted bundles on the RPi5
- [ ] Implement smoke inference validation with recent local production features
- [ ] Implement per-asset promotion logic with `current` and `previous`
- [ ] Update `active_bundles` state in `serving.duckdb` during promotion
- [ ] Record promotion failures and keep current bundle untouched when validation fails
- [ ] Add rollback workflow using `previous` pointers

### Phase 4: Application Backend

- [ ] Create FastAPI app structure for public, auth, private, and admin domains
- [ ] Add public prediction endpoints for latest predictions and history
- [ ] Add auth endpoints for login, logout, session introspection, and password change
- [ ] Add private portfolio endpoints for CRUD and analysis
- [ ] Add admin read-only ops endpoints
- [ ] Add health endpoints for API/runtime checks
- [ ] Add app config for `risk.manidmt.es` deployment assumptions

### Phase 5: Auth and Postgres

- [ ] Add Postgres service with persistent volume
- [ ] Add schema and migrations for `users`, `sessions`, `portfolios`, and `portfolio_positions`
- [ ] Implement email/password authentication
- [ ] Implement session cookies backed by Postgres
- [ ] Implement `admin` and `user` roles
- [ ] Implement forced password change on first login
- [ ] Implement CLI commands for bootstrap admin creation
- [ ] Implement CLI commands for manual user creation and password reset

### Phase 6: Public and Private Frontend

- [ ] Create one React app with `/`, `/app/*`, and `/ops/*` route zones
- [ ] Build public landing page
- [ ] Build public latest prediction views by asset
- [ ] Build public full historical prediction views by asset
- [ ] Build login and session-aware navigation
- [ ] Build private portfolio list and detail screens
- [ ] Build manual position entry with proxy assignment per row
- [ ] Build simple what-if portfolio UX
- [ ] Build admin-only `/ops` screens as read-only views
- [ ] Configure Vite build output for static serving by host Nginx

### Phase 7: RPi5 Deployment and Scheduling

- [ ] Add Dockerfiles for `api` and `worker`
- [ ] Add Compose setup for `api`, `worker`, and `postgres`
- [ ] Configure host Nginx to serve static frontend assets
- [ ] Configure host Nginx to proxy `/api` to FastAPI
- [ ] Add `systemd timer` and service units for the daily worker run
- [ ] Add deployment script or documented flow for app releases from the repo clone
- [ ] Add documented flow for weekly model release transfer to `incoming/`

### Phase 8: Operations, Backup, and Hardening

- [ ] Add structured logs for API and worker
- [ ] Surface key runtime health data into `/ops`
- [ ] Add daily backup for Postgres
- [ ] Add daily backup for `serving.duckdb`
- [ ] Add weekly backup for `financial_data.duckdb`
- [ ] Add backup retention policy for `current` and `previous` bundles
- [ ] Verify restore workflow for Postgres, DuckDBs, and bundles
- [ ] Run end-to-end smoke test on the RPi5 with real production configs
- [ ] Run failure drills for partial success, failed promotion, and rollback

### Phase 9: V1 Launch Readiness

- [ ] Confirm public/private/admin route behavior matches the final design
- [ ] Confirm one active 5-day bundle exists for each production asset
- [ ] Confirm historical predictions are being persisted correctly
- [ ] Confirm portfolio persistence and session handling work correctly
- [ ] Confirm `/ops` is admin-only and read-only
- [ ] Confirm app release and model release flows are operationally separate
- [ ] Confirm backup jobs have run successfully
- [ ] Freeze V1 scope and defer Phase 2 items explicitly

---

## 26. Repo-Oriented Implementation Plan

This section translates the V1 architecture into a concrete implementation path for the **current repo**, so the next steps are tied to real folders and modules rather than abstract phases only.

### 26.1 Recommended New Paths

These are the main new paths worth introducing.

Important boundary:

- `apps/*` should contain app entrypoints and UI/backend surfaces
- `src/quant_risk/prod/*` should contain the actual production business logic used by the RPi5 runtime
- `apps/*` should not become the home for core worker, inference, promotion, or portfolio domain logic

```text
apps/
  web/                        # React + Vite frontend
  api/                        # FastAPI app entrypoint

config/
  prod/
    datasources.rpi5.yaml
    features.rpi5.yaml
    assets.yaml

ops/
  docker/
    compose.rpi5.yml
  systemd/
    quant-risk-worker.service
    quant-risk-worker.timer
  nginx/
    risk.manidmt.es.conf

src/quant_risk/
  prod/
    assets.py
    bundle_manifest.py
    bundle_registry.py
    promotion.py
    schemas.py
    worker/
      main.py
      ingest.py
      features.py
      inference.py
      runs.py
    serving/
      duckdb.py
      predictions.py
      ops.py
    auth/
      models.py
      service.py
      sessions.py
      passwords.py
    portfolio/
      models.py
      service.py
      analysis.py

alembic/
  ...
```

### 26.2 Existing Modules to Reuse

The current repo already contains good building blocks. V1 should reuse these rather than reimplementing them.

- [`src/quant_risk/config.py`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/config.py): shared YAML/config helpers that should be extended instead of duplicated
- [`src/quant_risk/data/fetcher.py`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/data/fetcher.py): prices ingestion core
- [`src/quant_risk/data/macro.py`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/data/macro.py): macro ingestion core
- [`src/quant_risk/data/gkg.py`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/data/gkg.py): GKG ingestion core
- [`src/quant_risk/models/tabular/xgb.py`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/models/tabular/xgb.py): XGB load/inference path
- [`src/quant_risk/models/tabular/tabpfn.py`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/models/tabular/tabpfn.py): TabPFN load/inference path
- [`src/quant_risk/features/build.py`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/features/build.py): base feature engineering logic to mine for reusable per-asset production helpers

Existing CLI scripts should remain useful as references and smoke tools:

- [`scripts/ingest_prices.py`](/home/manidmt/TFM/quant-risk-tfm/scripts/ingest_prices.py)
- [`scripts/ingest_macro.py`](/home/manidmt/TFM/quant-risk-tfm/scripts/ingest_macro.py)
- [`scripts/ingest_gkg.py`](/home/manidmt/TFM/quant-risk-tfm/scripts/ingest_gkg.py)
- [`scripts/build_features.py`](/home/manidmt/TFM/quant-risk-tfm/scripts/build_features.py)

Important consequence:

- do **not** rebuild price ingestion from scratch
- do **not** rebuild macro ingestion from scratch
- do **not** rebuild GKG ingestion from scratch
- do **not** duplicate the generic YAML loading helpers
- do **not** rewrite XGB/TabPFN inference wrappers unless production needs a thin adapter layer
- do **not** clone the current feature builder wholesale when a production-specific extraction/adaptation is enough

### 26.3 Existing Paths That Should Stay Research-Oriented

These paths should **not** become the production runtime contract for the RPi5.

- [`scripts/walk_forward_chain_tab.py`](/home/manidmt/TFM/quant-risk-tfm/scripts/walk_forward_chain_tab.py): research/training/selection pipeline
- [`config/datasources.yaml`](/home/manidmt/TFM/quant-risk-tfm/config/datasources.yaml): current shared/research-oriented source config
- [`config/features.yaml`](/home/manidmt/TFM/quant-risk-tfm/config/features.yaml): current shared/research-oriented feature config
- [`src/quant_risk/datasets/make_dataset.py`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/datasets/make_dataset.py): training/evaluation dataset builder, not daily production serving logic
- [`src/quant_risk/features/labels.py`](/home/manidmt/TFM/quant-risk-tfm/src/quant_risk/features/labels.py): useful for training, not needed for daily RPi5 inference

The RPi5 worker should use production-specific modules under `src/quant_risk/prod/`, even if those modules call into existing lower-level ingestion/model code.

### 26.4 First Concrete Additions

If implementation starts now, this is the highest-leverage first file set.

#### Production config

- `config/prod/assets.yaml`
- `config/prod/datasources.rpi5.yaml`
- `config/prod/features.rpi5.yaml`

#### Bundle and serving contract

- `src/quant_risk/prod/assets.py`
- `src/quant_risk/prod/bundle_manifest.py`
- `src/quant_risk/prod/bundle_registry.py`
- `src/quant_risk/prod/schemas.py`

#### Worker skeleton and thin production adapters

- `src/quant_risk/prod/worker/main.py`
- `src/quant_risk/prod/worker/runs.py`

Only add these as thin orchestration/adaptation layers, not as rewrites of already existing logic:

- `src/quant_risk/prod/worker/ingest.py`
- `src/quant_risk/prod/worker/features.py`
- `src/quant_risk/prod/worker/inference.py`

#### Serving store helpers

- `src/quant_risk/prod/serving/duckdb.py`
- `src/quant_risk/prod/serving/predictions.py`
- `src/quant_risk/prod/serving/ops.py`

#### Web app bootstrap

- `apps/api/`
- `apps/web/`

### 26.5 Minimum Table Design to Define Early

#### `serving.duckdb`

Recommended first-pass tables:

- `active_bundles`
- `predictions_daily`
- `prediction_runs`
- `promotion_events`
- `asset_status`

Suggested minimum columns:

`active_bundles`
- `asset_id`
- `source_ticker`
- `release_id`
- `bundle_version`
- `model_type`
- `promoted_at`
- `previous_bundle_version`

`predictions_daily`
- `asset_id`
- `forecast_date`
- `predicted_class`
- `p_low`
- `p_medium`
- `p_high`
- `bundle_version`
- `data_cutoff_date`
- `status`
- `updated_at`

`prediction_runs`
- `run_id`
- `forecast_date`
- `asset_id`
- `attempt_no`
- `status`
- `error_code`
- `error_message`
- `started_at`
- `finished_at`

#### `Postgres`

Recommended first-pass tables:

- `users`
- `sessions`
- `portfolios`
- `portfolio_positions`

Suggested minimum columns:

`users`
- `id`
- `email`
- `password_hash`
- `role`
- `must_change_password`
- `is_active`
- `created_at`

`sessions`
- `id`
- `user_id`
- `expires_at`
- `created_at`
- `last_seen_at`

`portfolios`
- `id`
- `user_id`
- `name`
- `is_default`
- `created_at`
- `updated_at`

`portfolio_positions`
- `id`
- `portfolio_id`
- `label`
- `weight`
- `proxy_asset_id`
- `notes`
- `created_at`
- `updated_at`

### 26.6 Suggested Implementation Order in This Repo

This is the most practical order to start implementation without getting blocked by frontend polish too early.

1. Add `config/prod/*` and the asset catalog.
2. Define the bundle manifest contract and `serving.duckdb` tables.
3. Build the production worker entrypoint and run orchestration under `src/quant_risk/prod/worker/`.
4. Wrap/reuse the existing ingestion modules from `src/quant_risk/data/*`.
5. Extract/adapt a production-specific feature path from `src/quant_risk/features/build.py` that removes cross-asset dependencies.
6. Add bundle loading and per-asset inference on top of the existing model wrappers.
7. Persist latest and historical predictions to `serving.duckdb`.
8. Add promotion and rollback helpers.
9. Add Postgres schema + auth/session layer.
10. Add portfolio storage and backend analysis service.
11. Add FastAPI routes.
12. Add React frontend.
13. Add Nginx, Compose, and `systemd` integration.
14. Run end-to-end smoke tests on the RPi5.

### 26.7 Recommended New Tests

The first production-specific tests should live separately from current research tests and should focus on new production contracts rather than duplicating ingest/model smoke coverage that already exists.

Recommended additions:

- `tests/prod/test_asset_catalog.py`
- `tests/prod/test_bundle_manifest.py`
- `tests/prod/test_bundle_registry.py`
- `tests/prod/test_worker_orchestration.py`
- `tests/prod/test_worker_features_per_asset.py`
- `tests/prod/test_bundle_inference_contract.py`
- `tests/prod/test_serving_duckdb_contract.py`
- `tests/prod/test_auth_session_flow.py`
- `tests/prod/test_portfolio_analysis.py`
- `tests/prod/test_promotion_current_previous.py`

### 26.8 Practical Rule of Thumb

When in doubt:

- add production logic under `src/quant_risk/prod/`
- keep training/research logic where it already lives
- reuse low-level ingestion/model helpers
- wrap or adapt existing code before rewriting it
- do not let the RPi5 runtime depend directly on research scripts as its public contract

## 27. Web Visual Design

This section defines the first-pass visual direction for `risk.manidmt.es`.

The goal for V1 is not a flashy fintech landing page. It should feel minimal, calm, serious, and personal: a polished master's thesis project that already behaves like a real product.

### 27.1 Visual Direction

Recommended visual identity:

- minimal and editorial rather than startup-like
- light, calm, off-white background rather than bright white
- strong typography with a subtle academic feel
- restrained use of color
- generous whitespace
- motion that feels atmospheric rather than decorative

The public landing page should feel quieter and more personal than the authenticated product surfaces. The visual tone should suggest: "research converted into a polished tool".

### 27.2 Core Aesthetic Principles

The design should follow these rules:

- avoid loud gradients, neon accents, or "crypto" aesthetics
- avoid dashboard overload on the landing page
- keep the number of visible colors low
- use color mainly for information, not decoration
- prefer typography, spacing, and composition over heavy UI chrome
- keep background motion very subtle and slow

### 27.3 Landing Page Structure

The landing page at `/` should be intentionally sparse.

#### Header

The header should be minimal and balanced:

- left: project wordmark or short project name
- center: `Predictions` and `Portfolio`
- right:
  - `Log in` or `Sign in` when the user is not authenticated
  - nothing when the user is already authenticated

Behavior:

- `Predictions` should be publicly accessible
- `Portfolio` should remain visible in the navigation even when logged out
- if a logged-out user clicks `Portfolio`, they should be redirected to login

#### Hero

The hero should be the primary visual focus of the page and remain extremely clean.

Recommended hierarchy:

- time-based greeting:
  - `Good morning`
  - `Good afternoon`
  - `Good evening`
- if the user is authenticated, show their name directly underneath
- project title
- small line reading `Master's Thesis`
- one short descriptive sentence

Example structure:

- greeting
- optional user name
- project title
- `Master's Thesis`
- one-line explanation

#### Primary Actions

Below the hero, the landing page should expose two clean access points:

- `View predictions`
- `Open portfolio`

These should not look like loud marketing buttons. They can be rendered as understated panels, large text links, or soft bordered action blocks.

#### Footer

The footer should remain minimal.

Recommended content:

- `About the author`
- one or two lines of context
- optional links to GitHub, LinkedIn, or personal website

### 27.4 Visual Tone by Route

The single React app should still differentiate visual density by route.

`/`
- most minimal and editorial
- highest whitespace
- most personal tone

`/app/*`
- more product-oriented
- cleaner dashboard patterns
- still restrained and light

`/ops/*`
- denser and more utilitarian
- minimal decoration
- clear status-first layout

### 27.5 Typography

Recommended font stack:

- `Newsreader` for the landing hero and major title moments
- `IBM Plex Sans` for navigation, paragraphs, forms, and general UI
- `IBM Plex Mono` for dates, probabilities, labels, and tabular numeric content

Rationale:

- `Newsreader` adds an editorial and thesis-like character
- `IBM Plex Sans` keeps the interface technical and readable
- `IBM Plex Mono` provides a precise quantitative feel for data

Recommended hierarchy:

- greeting: small `IBM Plex Sans`
- optional user name: medium `Newsreader`
- project title: large `Newsreader`
- `Master's Thesis`: small uppercase `IBM Plex Sans` or `IBM Plex Mono`
- body and navigation: `IBM Plex Sans`
- data labels and probabilities: `IBM Plex Mono`

### 27.6 Color Palette

The visual palette should be built around an off-white professional background and muted dark blue-gray text.

Suggested design tokens:

```css
:root {
  --bg: #f7f4ee;
  --bg-soft: #f1ece3;
  --surface: rgba(255, 255, 255, 0.72);

  --text: #18212b;
  --text-soft: #5f6b76;
  --text-faint: #8a938f;

  --line: #d9d3c8;
  --line-strong: #c2c9cf;

  --accent: #21384d;
  --accent-soft: #6b8298;

  --low: #4f7a64;
  --medium: #a57a2a;
  --high: #9a5246;
}
```

Recommended usage:

- `--bg` as the main application background
- `--bg-soft` for subtle sections or bands
- `--surface` for cards and light containers
- `--text` for the main content
- `--accent` for links, navigation emphasis, and subtle interactive elements
- `--low`, `--medium`, and `--high` only for prediction signals and risk states

Important constraint:

- the signal colors should not become the decorative brand palette
- they should remain mostly reserved for actual prediction meaning

### 27.7 Layout and Spacing Tokens

The design should rely on spacing and rhythm more than borders or shadows.

Suggested layout tokens:

```css
:root {
  --radius-sm: 10px;
  --radius-md: 18px;
  --radius-lg: 28px;

  --space-1: 8px;
  --space-2: 12px;
  --space-3: 16px;
  --space-4: 24px;
  --space-5: 32px;
  --space-6: 48px;
  --space-7: 72px;
  --space-8: 96px;

  --max-width: 1120px;
}
```

Use:

- wide horizontal breathing room
- large hero spacing
- light sections with soft grouping
- soft radius values rather than sharp corners

### 27.8 Background Motion

The background may include subtle animated traces inspired by market charts or moving line work, but the effect must remain restrained.

Recommended behavior:

- extremely low opacity
- thin line work
- slow movement
- no flashy loops
- no bright glow effects

Good references in spirit:

- faint chart-like lines
- moving contour paths
- soft grid or path overlays

Bad directions:

- particle-heavy backgrounds
- neon candlestick motifs
- obvious "AI" or "crypto" visuals
- anything that competes with the hero text

### 27.9 Interaction and Motion

Motion should support calmness and clarity.

Recommended interaction patterns:

- gentle fade-and-slide entrance for hero content
- restrained hover states
- soft underline or color shift for links
- short transitions in the `160ms-220ms` range

Avoid:

- spring-heavy animations
- overshoot/bounce
- complex motion choreography on the landing page

### 27.10 Summary

The intended visual identity for V1 can be summarized as:

- minimal
- off-white
- editorial
- academically polished
- technically credible
- quiet rather than flashy

If the product grows later with chat, richer dashboards, and admin tooling, the visual system should still keep this same foundation instead of pivoting to a completely different brand language.

---

## 28. Future Work

Features explicitly deferred from V1 to keep scope manageable.

### 28.1 Portfolio: cash as implicit residual position

**Design decision (V1):** When portfolio weights do not sum to 100%, the remainder is treated as cash — an asset with zero volatility. This is handled implicitly by the analysis service when normalising weights. It is not shown as an explicit position in the UI, but its effect is that under-allocated portfolios are effectively assumed to hold the difference in cash.

**Example:** a portfolio with 50% AAPL and 30% BTC has 20% implicit cash. The weighted portfolio signal reflects this: cash contributes zero to `p_high`, pulling the blended signal towards low volatility.

**Future improvement:** expose the implicit cash position in the analysis result so the user can see its contribution explicitly.

### 28.2 Portfolio: real-unit positions (V2)

**Status:** Designed, deferred from V1 due to added complexity.

In V1 positions are stored as `weight_pct` (approximate percentage, normalised internally by the analysis service). In V2 the preferred model is for users to enter the actual number of units/shares held for each asset, with the app computing percentage weights automatically using real-time market prices.

**Changes required for V2:**
- New `units` field on `PortfolioPosition` (DB migration required)
- Real-time price endpoint per `proxy_asset_id` (or enrich `/api/public/prices/history`)
- Frontend: "Units" input instead of "Weight %", dynamic portfolio weight breakdown
- Remove implicit normalisation from the analysis service
