'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-04-09

@description: Off-box prediction pusher — workstation → RPi5.

Runs the full daily inference pipeline for every production asset on the
workstation and POSTs each resulting prediction to the RPi5 API. The RPi5
persists them via POST /api/internal/predictions.

Why this exists
---------------
TabPFN inference on the RPi5 CPU takes ~1–2 hours per asset and keeps hitting
memory / CUDA-stub edge cases.  Offloading inference to the workstation makes
the RPi5 a pure read/serve box and cuts the daily batch to a few minutes.

Usage
-----
    # Mandatory env vars
    export QUANT_RISK_RPI5_URL=https://risk.manidmt.es
    export QUANT_RISK_INTERNAL_TOKEN=<same token as on RPi5 api container>

    # Dry-run (run inference, print payloads, skip POSTs)
    python scripts/push_predictions_to_rpi5.py --dry-run

    # Real run for yesterday (D-1)
    python scripts/push_predictions_to_rpi5.py

    # Override forecast date
    python scripts/push_predictions_to_rpi5.py --forecast-date 2026-04-07

    # Limit to specific assets
    python scripts/push_predictions_to_rpi5.py --assets us_equities bitcoin

The script reuses the production worker's orchestration
(``quant_risk.prod.worker.main.run_asset_batch``) but replaces the persist step
with an HTTP POST.  Ingest / feature build / inference all run locally against
whatever ``financial_data.duckdb`` and ``bundles_dir`` the host provides.
'''

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import urllib.error
import urllib.request

import duckdb

from quant_risk.prod.assets import AssetInfo, load_asset_catalog
from quant_risk.prod.schemas import RunStatus
from quant_risk.prod.ticker_universe import load_ticker_universe
from quant_risk.prod.worker.ingest import (
    IngestResult,
    SharedIngestResult,
    ingest_for_asset,
    run_shared_ingest,
)
from quant_risk.prod.worker.features import build_features_for_asset
from quant_risk.prod.worker.inference import PredictionOutput, run_inference_for_asset

logger = logging.getLogger("push_predictions_to_rpi5")

_DEFAULT_TIMEOUT_SEC = 30


# ---------------------------------------------------------------------------
# HTTP client — stdlib only to avoid extra deps on the research machine.
# ---------------------------------------------------------------------------

@dataclass
class Rpi5Client:
    base_url: str
    token: str
    timeout: float = _DEFAULT_TIMEOUT_SEC

    def _post(self, path: str, data: bytes) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "push_predictions_to_rpi5/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RPi5 responded {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"RPi5 unreachable ({exc.reason}).") from exc

    def push_prediction(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/internal/predictions", json.dumps(payload).encode("utf-8"))

    def push_prices_bulk(self, rows: list[dict[str, Any]]) -> int:
        data = json.dumps({"rows": rows}).encode("utf-8")
        resp = self._post("/api/internal/prices/bulk", data)
        return int(resp.get("upserted", len(rows)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _determine_run_status(ingest_result: IngestResult) -> RunStatus:
    if not ingest_result.prices_ok or not ingest_result.macro_ok:
        return RunStatus.stale_data
    if not ingest_result.news_ok:
        return RunStatus.stale_news
    return RunStatus.fresh


def _build_payload(
    asset: AssetInfo,
    forecast_date: date,
    prediction: PredictionOutput,
    ingest_result: IngestResult,
    run_status: RunStatus,
    source_host: str,
) -> dict[str, Any]:
    return {
        "asset_id": asset.asset_id,
        "forecast_date": forecast_date.isoformat(),
        "predicted_class": prediction.predicted_class.value,
        "p_low": float(prediction.p_low),
        "p_medium": float(prediction.p_medium),
        "p_high": float(prediction.p_high),
        "bundle_version": prediction.bundle_version,
        "data_cutoff_date": ingest_result.data_cutoff_date.isoformat(),
        "run_status": run_status.value,
        "source_host": source_host,
    }


def _process_asset(
    asset: AssetInfo,
    datasources_config: str,
    features_config: str,
    bundles_dir: str,
    financial_db_path: str | None,
    shared: SharedIngestResult | None,
    forecast_date: date,
    client: Rpi5Client | None,
    source_host: str,
    dry_run: bool,
) -> tuple[bool, str]:
    """Run the full pipeline for one asset and push the prediction.

    Returns
    -------
    (ok, detail)
        ``ok`` is True on success (push accepted, or dry-run completed).
        ``detail`` is a short human string for the batch summary.
    """
    logger.info("[%s] starting forecast_date=%s", asset.asset_id, forecast_date)

    try:
        ingest_result = ingest_for_asset(
            asset_id=asset.asset_id,
            source_ticker=asset.source_ticker,
            datasources_config=datasources_config,
            db_path=financial_db_path,
            shared_ingest=shared,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] ingest failed: %s", asset.asset_id, exc)
        return False, f"ingest: {exc}"

    # Load feature contract from the active bundle (if present) for better alignment.
    feature_contract: list[str] | None = None
    try:
        from quant_risk.prod.bundle_registry import get_active_bundle_dir
        _bd = get_active_bundle_dir(bundles_dir, asset.asset_id)
        _fc = _bd / "feature_contract.json"
        if _fc.exists():
            feature_contract = json.loads(_fc.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("[%s] no feature_contract from bundle (%s)", asset.asset_id, exc)

    try:
        features_result = build_features_for_asset(
            asset_id=asset.asset_id,
            source_ticker=asset.source_ticker,
            ingest_result=ingest_result,
            features_config=features_config,
            db_path=financial_db_path,
            feature_contract=feature_contract,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] feature build failed: %s", asset.asset_id, exc)
        return False, f"features: {exc}"

    try:
        prediction: PredictionOutput = run_inference_for_asset(
            asset_id=asset.asset_id,
            features=features_result.features,
            bundles_dir=bundles_dir,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] inference failed: %s", asset.asset_id, exc)
        return False, f"inference: {exc}"

    run_status = _determine_run_status(ingest_result)
    payload = _build_payload(
        asset=asset,
        forecast_date=forecast_date,
        prediction=prediction,
        ingest_result=ingest_result,
        run_status=run_status,
        source_host=source_host,
    )

    if dry_run or client is None:
        logger.info("[%s] DRY-RUN payload=%s", asset.asset_id, json.dumps(payload))
        return True, f"dry-run {prediction.predicted_class.value}"

    try:
        resp = client.push_prediction(payload)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] push failed: %s", asset.asset_id, exc)
        return False, f"push: {exc}"

    logger.info(
        "[%s] pushed — class=%s attempt_no=%s stored_at=%s",
        asset.asset_id,
        prediction.predicted_class.value,
        resp.get("attempt_no"),
        resp.get("stored_at"),
    )
    return True, f"ok {prediction.predicted_class.value}"


# ---------------------------------------------------------------------------
# Price push
# ---------------------------------------------------------------------------

_PRICES_BATCH_SIZE = 2_000
_PRICES_WINDOW_DAYS = 504  # ~2 years, matching risk_adjustment.py default


def _push_prices(
    client: Rpi5Client,
    financial_db_path: str,
    universe_catalog: str,
    window_days: int = _PRICES_WINDOW_DAYS,
    batch_size: int = _PRICES_BATCH_SIZE,
) -> tuple[int, int]:
    """Read raw_prices for the full ticker universe from the local DB and push to RPi5.

    Returns
    -------
    (total_rows, failed_batches)
    """
    universe = load_ticker_universe(universe_catalog)
    tickers = list({t.ticker for t in universe})
    if not tickers:
        logger.warning("Ticker universe is empty — skipping price push.")
        return 0, 0

    logger.info("Pushing raw_prices for %d universe tickers (window=%d days)", len(tickers), window_days)

    in_list = ", ".join(f"'{t}'" for t in tickers)
    try:
        conn = duckdb.connect(financial_db_path, read_only=True)
        rows = conn.execute(f"""
            SELECT ticker,
                   date::VARCHAR AS date,
                   open,
                   high,
                   low,
                   close,
                   volume
            FROM raw_prices
            WHERE ticker IN ({in_list})
              AND date >= CURRENT_DATE - INTERVAL ({window_days} || ' days')
              AND close IS NOT NULL
            ORDER BY ticker, date
        """).fetchall()
        conn.close()
    except Exception as exc:
        logger.exception("Failed to read raw_prices from local DB: %s", exc)
        return 0, 1

    if not rows:
        logger.warning("No raw_prices rows found for universe tickers in window.")
        return 0, 0

    logger.info("Read %d price rows from local DB; pushing in batches of %d", len(rows), batch_size)

    total_pushed = 0
    failed_batches = 0
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    for idx, batch in enumerate(batches, 1):
        payload = [
            {
                "ticker": r[0],
                "date": r[1],
                "open": r[2],
                "high": r[3],
                "low": r[4],
                "close": r[5],
                "volume": r[6],
            }
            for r in batch
        ]
        try:
            n = client.push_prices_bulk(payload)
            total_pushed += n
            logger.info("Prices batch %d/%d — pushed %d rows", idx, len(batches), n)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prices batch %d/%d failed: %s", idx, len(batches), exc)
            failed_batches += 1

    return total_pushed, failed_batches


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="push_predictions_to_rpi5",
        description="Run the daily inference pipeline locally and push predictions to the RPi5.",
    )
    p.add_argument("--assets-config", default="config/prod/assets.yaml")
    p.add_argument("--datasources-config", default="config/prod/datasources.rpi5.yaml")
    p.add_argument("--features-config", default="config/prod/features.rpi5.yaml")
    p.add_argument(
        "--bundles-dir",
        default=os.environ.get("QUANT_RISK_BUNDLES_DIR", "bundles"),
        help="Root of the bundle tree (defaults to $QUANT_RISK_BUNDLES_DIR or ./bundles).",
    )
    p.add_argument(
        "--financial-db",
        default=os.environ.get("QUANT_RISK_DB_PATH"),
        help="Path to financial_data.duckdb (defaults to $QUANT_RISK_DB_PATH).",
    )
    p.add_argument(
        "--forecast-date",
        default=None,
        help="Override forecast date (YYYY-MM-DD). Defaults to D-1.",
    )
    p.add_argument(
        "--assets",
        nargs="+",
        default=None,
        help="Limit to this subset of asset_ids (default: all).",
    )
    p.add_argument(
        "--rpi5-url",
        default=os.environ.get("QUANT_RISK_RPI5_URL"),
        help="Base URL of the RPi5 API (default: $QUANT_RISK_RPI5_URL).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT_SEC,
        help="HTTP timeout per push in seconds.",
    )
    p.add_argument(
        "--skip-shared-ingest",
        action="store_true",
        help="Skip the macro + GKG shared ingest (use existing DB state).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference but do not push to the RPi5 (print payloads instead).",
    )
    p.add_argument(
        "--skip-prices",
        action="store_true",
        help="Skip pushing raw_prices for the ticker universe (prices are pushed by default).",
    )
    p.add_argument(
        "--universe-catalog",
        default="config/prod/ticker_universe.yaml",
        help="Path to ticker_universe.yaml used for the price push (default: %(default)s).",
    )
    p.add_argument(
        "--prices-window-days",
        type=int,
        default=_PRICES_WINDOW_DAYS,
        help="Calendar-day lookback window for price rows (default: %(default)s).",
    )
    p.add_argument(
        "--prices-batch-size",
        type=int,
        default=_PRICES_BATCH_SIZE,
        help="Rows per HTTP batch for the price push (default: %(default)s).",
    )
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    forecast_date = (
        date.fromisoformat(args.forecast_date)
        if args.forecast_date
        else date.today() - timedelta(days=1)
    )

    # --- Resolve client (None on dry-run) --------------------------------
    client: Rpi5Client | None = None
    if not args.dry_run:
        token = os.environ.get("QUANT_RISK_INTERNAL_TOKEN")
        if not args.rpi5_url:
            logger.error("Missing --rpi5-url / $QUANT_RISK_RPI5_URL.")
            return 2
        if not token:
            logger.error("Missing $QUANT_RISK_INTERNAL_TOKEN.")
            return 2
        client = Rpi5Client(
            base_url=args.rpi5_url,
            token=token,
            timeout=float(args.timeout),
        )
        logger.info("Target RPi5: %s", args.rpi5_url)
    else:
        logger.info("Dry run — no HTTP calls will be made.")

    # --- Load asset catalog ---------------------------------------------
    catalog = load_asset_catalog(args.assets_config)
    if args.assets:
        wanted = set(args.assets)
        catalog = [a for a in catalog if a.asset_id in wanted]
        missing = wanted - {a.asset_id for a in catalog}
        if missing:
            logger.error("Unknown asset_ids requested: %s", sorted(missing))
            return 2
    if not catalog:
        logger.error("No assets to process after filtering.")
        return 2

    logger.info(
        "Processing %d assets: %s",
        len(catalog), [a.asset_id for a in catalog],
    )

    # --- Shared ingest ---------------------------------------------------
    shared: SharedIngestResult | None = None
    if not args.skip_shared_ingest:
        try:
            shared = run_shared_ingest(
                datasources_config=args.datasources_config,
                db_path=args.financial_db,
            )
            logger.info(
                "Shared ingest done — macro_ok=%s news_ok=%s",
                shared.macro_ok, shared.news_ok,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Shared ingest failed: %s — continuing.", exc)
    else:
        logger.info("Skipping shared ingest (--skip-shared-ingest).")

    # --- Per-asset loop --------------------------------------------------
    source_host = socket.gethostname()
    successes: list[str] = []
    failures: list[tuple[str, str]] = []
    for asset in catalog:
        ok, detail = _process_asset(
            asset=asset,
            datasources_config=args.datasources_config,
            features_config=args.features_config,
            bundles_dir=args.bundles_dir,
            financial_db_path=args.financial_db,
            shared=shared,
            forecast_date=forecast_date,
            client=client,
            source_host=source_host,
            dry_run=args.dry_run,
        )
        if ok:
            successes.append(f"{asset.asset_id}={detail}")
        else:
            failures.append((asset.asset_id, detail))

    logger.info(
        "Batch summary — forecast_date=%s succeeded=%d %s failed=%d %s",
        forecast_date,
        len(successes), successes,
        len(failures), failures,
    )

    # --- Price push (universe tickers for risk adjustment) ----------------
    price_failures = 0
    if not args.dry_run and not args.skip_prices and client is not None:
        db_path = args.financial_db or os.environ.get("QUANT_RISK_DB_PATH")
        if not db_path:
            logger.warning(
                "Skipping price push — no DB path (set --financial-db or $QUANT_RISK_DB_PATH)."
            )
        else:
            pushed, price_failures = _push_prices(
                client=client,
                financial_db_path=db_path,
                universe_catalog=args.universe_catalog,
                window_days=args.prices_window_days,
                batch_size=args.prices_batch_size,
            )
            logger.info("Price push complete — rows=%d failed_batches=%d", pushed, price_failures)
    elif args.dry_run:
        logger.info("Dry run — skipping price push.")

    return 0 if not failures and price_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
