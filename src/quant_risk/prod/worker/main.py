'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Production worker entrypoint — daily batch orchestration.

This module is the single entry point for the RPi5 daily batch.  It is
launched by a systemd timer (rpi5.md §10.1) and exits after completing
the batch.  It does NOT run as a persistent server.

Batch flow per asset (rpi5.md §10.3):
    1. Ingest prices, macro, and GKG/news data  ← worker/ingest.py (Step 4)
    2. Build production feature vector          ← worker/features.py (Step 5)
    3. Load active bundle + run inference       ← worker/inference.py (Step 6)
    4. Persist prediction to serving.duckdb     ← serving/predictions.py (Step 7)
    5. Update operational status                ← worker/runs.py

Failure policy (rpi5.md §10.4):
    - Partial success by asset: one asset failing never aborts the others.
    - RunStatus per asset: fresh | stale_news | stale_data | failed.
    - At most one automatic retry per asset for transient failures.

Exit codes:
    0  — at least one asset produced a non-failed prediction
    1  — all assets failed

Reference: rpi5.md §10, §13.2, §26.1, §26.4
'''

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

import json

from quant_risk.prod.assets import AssetInfo, load_asset_catalog
from quant_risk.prod.bundle_registry import get_active_bundle_dir
from quant_risk.prod.schemas import RunStatus
from quant_risk.prod.serving.duckdb import ServingDB
from quant_risk.prod.worker.ingest import (
    IngestResult,
    SharedIngestResult,
    ingest_for_asset,
    run_shared_ingest,
)
from quant_risk.prod.worker.features import FeaturesResult, build_features_for_asset
from quant_risk.prod.worker.inference import PredictionOutput, run_inference_for_asset
from quant_risk.prod.serving.predictions import persist_prediction
from quant_risk.prod.worker.runs import (
    RunContext,
    RunResult,
    finish_run,
    make_failed_result,
    start_run,
    update_asset_status,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WorkerConfig:
    """Runtime configuration for one daily batch execution.

    All paths default to the RPi5 production layout (rpi5.md §22).
    Override via CLI arguments or environment variables as needed.
    """

    assets_config: str = "config/prod/assets.yaml"
    datasources_config: str = "config/prod/datasources.rpi5.yaml"
    features_config: str = "config/prod/features.rpi5.yaml"
    bundles_dir: str = "/srv/quant-risk/bundles"
    serving_db_path: str | None = None   # path to serving.duckdb (env var fallback)
    financial_db_path: str | None = None  # path to financial_data.duckdb (env var / config fallback)
    # D-1 forecast date: None means "use yesterday" (determined at runtime).
    forecast_date: date | None = None
    # Maximum automatic retries per asset on transient failures (rpi5.md §10.5).
    max_retries: int = 1
    # Dry-run: go through orchestration logic but skip all I/O steps.
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_forecast_date(cfg: WorkerConfig) -> date:
    """Return the forecast date to use for this batch run.

    Defaults to D-1 (yesterday) per rpi5.md §10.2.
    """
    return cfg.forecast_date or (date.today() - timedelta(days=1))


def _determine_run_status(ingest_result: IngestResult) -> RunStatus:
    """Map IngestResult availability flags to a RunStatus code."""
    if not ingest_result.prices_ok or not ingest_result.macro_ok:
        return RunStatus.stale_data
    if not ingest_result.news_ok:
        return RunStatus.stale_news
    return RunStatus.fresh


# ---------------------------------------------------------------------------
# Per-asset pipeline
# ---------------------------------------------------------------------------

def run_asset_batch(
    asset: AssetInfo,
    config: WorkerConfig,
    db: ServingDB,
    forecast_date: date,
    attempt_no_override: int | None = None,
    shared_ingest: SharedIngestResult | None = None,
    *,
    # Injectable step functions — used in tests to replace real implementations.
    _ingest_fn: Callable = ingest_for_asset,
    _features_fn: Callable = build_features_for_asset,
    _inference_fn: Callable = run_inference_for_asset,
    _persist_fn: Callable = persist_prediction,
) -> RunResult:
    """Execute the full production pipeline for one asset on one forecast date.

    Parameters
    ----------
    asset:
        The production asset to process.
    config:
        Worker configuration (paths, flags).
    db:
        Open ServingDB connection.
    forecast_date:
        The D-1 date for which to produce a prediction.
    attempt_no_override:
        If set, force this attempt number (used in tests).
    _*_fn:
        Injectable step callables — allows unit-testing the orchestration
        without a live database or real model artefacts.

    Returns
    -------
    RunResult
        Always returns (never raises), even on failure.
    """
    # dry_run exits before start_run so no DB rows are written at all.
    if config.dry_run:
        logger.info("[%s] dry_run=True — skipping all I/O steps.", asset.asset_id)
        return RunResult(status=RunStatus.fresh, bundle_version="dry_run")

    ctx: RunContext = start_run(db, asset.asset_id, forecast_date)
    result: RunResult

    try:
        logger.info(
            "[%s] Starting attempt %d for forecast_date=%s",
            asset.asset_id, ctx.attempt_no, forecast_date,
        )

        # Step 1: Ingest prices (macro/GKG shared result passed from batch level)
        ingest_result: IngestResult = _ingest_fn(
            asset_id=asset.asset_id,
            source_ticker=asset.source_ticker,
            datasources_config=config.datasources_config,
            db_path=config.financial_db_path,
            shared_ingest=shared_ingest,
        )

        # Step 2: Features — load feature_contract from the active bundle for
        # missing-column ops visibility; safe to skip if bundle not yet present.
        feature_contract: list[str] | None = None
        try:
            _bd = get_active_bundle_dir(config.bundles_dir, asset.asset_id)
            _fc_path = _bd / "feature_contract.json"
            if _fc_path.exists():
                feature_contract = json.loads(_fc_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass

        features_result: FeaturesResult = _features_fn(
            asset_id=asset.asset_id,
            source_ticker=asset.source_ticker,
            ingest_result=ingest_result,
            features_config=config.features_config,
            db_path=config.financial_db_path,
            feature_contract=feature_contract,
        )

        # Step 3: Inference
        prediction: PredictionOutput = _inference_fn(
            asset_id=asset.asset_id,
            features=features_result.features,
            bundles_dir=config.bundles_dir,
        )

        # Step 4: Persist
        run_status = _determine_run_status(ingest_result)
        _persist_fn(
            db=db,
            asset_id=asset.asset_id,
            forecast_date=forecast_date,
            prediction=prediction,
            status=run_status,
            data_cutoff_date=ingest_result.data_cutoff_date,
        )

        result = RunResult(
            status=run_status,
            bundle_version=prediction.bundle_version,
        )
        logger.info(
            "[%s] Completed — status=%s, bundle=%s",
            asset.asset_id, result.status.value, result.bundle_version,
        )

    except NotImplementedError as exc:
        result = make_failed_result(
            error_message=str(exc),
            error_code="NOT_IMPLEMENTED",
        )
        logger.error("[%s] Not implemented: %s", asset.asset_id, exc)

    except Exception as exc:  # noqa: BLE001
        result = make_failed_result(
            error_message=str(exc),
            error_code=type(exc).__name__,
        )
        logger.exception("[%s] Unexpected error: %s", asset.asset_id, exc)

    finally:
        finish_run(db, ctx, result)
        update_asset_status(db, ctx, result)

    return result


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def run_daily_batch(config: WorkerConfig) -> dict[str, RunStatus]:
    """Run the daily batch for all production assets.

    Partial success semantics (rpi5.md §10.4):
    - Each asset is processed independently.
    - A failure for one asset never aborts the others.
    - Automatic retry up to config.max_retries for failed assets.

    Returns
    -------
    dict[str, RunStatus]
        Mapping of asset_id → final RunStatus for this batch run.
    """
    catalog = load_asset_catalog(config.assets_config)
    forecast_date = _resolve_forecast_date(config)

    logger.info(
        "Starting daily batch — forecast_date=%s, assets=%d, dry_run=%s",
        forecast_date, len(catalog), config.dry_run,
    )

    db = ServingDB(config.serving_db_path)
    db.create_schema()

    final_statuses: dict[str, RunStatus] = {}

    try:
        # Phase 1: shared ingest (macro + GKG) — run ONCE for all assets.
        shared: SharedIngestResult | None = None
        if not config.dry_run:
            try:
                shared = run_shared_ingest(
                    datasources_config=config.datasources_config,
                    db_path=config.financial_db_path,
                )
                logger.info(
                    "Shared ingest complete — macro_ok=%s, news_ok=%s",
                    shared.macro_ok, shared.news_ok,
                )
            except Exception as exc:
                logger.exception("Shared ingest failed: %s — continuing per-asset.", exc)

        # Phase 2: per-asset loop (prices + features + inference + persist).
        for asset in catalog:
            # Pass module-level names explicitly so monkeypatching in tests works.
            step_kwargs = dict(
                _ingest_fn=ingest_for_asset,
                _features_fn=build_features_for_asset,
                _inference_fn=run_inference_for_asset,
                _persist_fn=persist_prediction,
            )
            result = run_asset_batch(
                asset, config, db, forecast_date,
                shared_ingest=shared, **step_kwargs,
            )

            # Automatic retry on transient failure (rpi5.md §10.5).
            if result.status == RunStatus.failed and config.max_retries > 0:
                logger.warning(
                    "[%s] Attempt 1 failed (%s) — retrying once.",
                    asset.asset_id, result.error_code,
                )
                result = run_asset_batch(
                    asset, config, db, forecast_date,
                    shared_ingest=shared, **step_kwargs,
                )

            final_statuses[asset.asset_id] = result.status

    finally:
        db.close()

    _log_batch_summary(forecast_date, final_statuses)
    return final_statuses


def _log_batch_summary(
    forecast_date: date,
    statuses: dict[str, RunStatus],
) -> None:
    succeeded = [a for a, s in statuses.items() if s != RunStatus.failed]
    failed = [a for a, s in statuses.items() if s == RunStatus.failed]

    logger.info(
        "Batch summary — forecast_date=%s | succeeded=%d %s | failed=%d %s",
        forecast_date,
        len(succeeded), succeeded,
        len(failed), failed,
    )


# ---------------------------------------------------------------------------
# CLI entry point (launched by systemd timer)
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="quant-risk-worker",
        description="quant-risk daily production worker (rpi5.md §10, §13.2)",
    )
    p.add_argument("--assets-config",      default="config/prod/assets.yaml")
    p.add_argument("--datasources-config", default="config/prod/datasources.rpi5.yaml")
    p.add_argument("--features-config",    default="config/prod/features.rpi5.yaml")
    p.add_argument("--bundles-dir",        default="/srv/quant-risk/bundles")
    p.add_argument("--serving-db",         default=None, dest="serving_db_path",
                   help="Path to serving.duckdb. Falls back to QUANT_RISK_SERVING_DB_PATH.")
    p.add_argument("--forecast-date",      default=None,
                   help="Override forecast date (YYYY-MM-DD). Defaults to D-1.")
    p.add_argument("--max-retries",        type=int, default=1)
    p.add_argument("--dry-run",            action="store_true",
                   help="Orchestrate without executing I/O steps.")
    p.add_argument("--log-level",          default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    forecast_date: date | None = None
    if args.forecast_date:
        forecast_date = date.fromisoformat(args.forecast_date)

    config = WorkerConfig(
        assets_config=args.assets_config,
        datasources_config=args.datasources_config,
        features_config=args.features_config,
        bundles_dir=args.bundles_dir,
        serving_db_path=args.serving_db_path,
        forecast_date=forecast_date,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
    )

    statuses = run_daily_batch(config)

    any_success = any(s != RunStatus.failed for s in statuses.values())
    sys.exit(0 if any_success else 1)
