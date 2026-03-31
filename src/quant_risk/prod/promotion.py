'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Bundle promotion and rollback helpers (rpi5.md §9).

Promotion is **independent by asset** — one asset failing never blocks others.

Promotion flow per asset (rpi5.md §9):
  1. Validate the candidate bundle (manifest integrity + model load + checksums)
  2. If validation passes:
       a. Repoint `previous` → old `current` target
       b. Repoint `current`  → new bundle version
       c. Update `active_bundles` in serving.duckdb
       d. Record a `promoted` event in `promotion_events`
  3. If validation fails:
       a. Leave `current` untouched
       b. Record a `failed` event in `promotion_events`

Rollback flow per asset:
  1. `previous` pointer must exist
  2. Swap `current` ← `previous` target
  3. Update `active_bundles` in serving.duckdb to the previous bundle
  4. Record a `promoted` event (bundle_version = previous version)

Pointer convention:
  `current` and `previous` are symlinks pointing to bundle version directories.
  Symlinks are the canonical convention on the RPi5 (Linux/ext4).
  Plain text-file pointers (written by _write_pointer) are also supported for
  environments where symlinks are unavailable (e.g. Windows CI).

Reference: rpi5.md §8.4, §9
'''

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_risk.prod.bundle_manifest import BundleManifest
from quant_risk.prod.schemas import PromotionStatus
from quant_risk.prod.serving.duckdb import ServingDB
from quant_risk.prod.worker.inference import _load_model_artifact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of validating one candidate bundle.

    Attributes
    ----------
    passed:
        True if all validation checks succeeded.
    bundle_version:
        The bundle version string from manifest.json.
    errors:
        List of error messages for any failed checks (empty when passed=True).
    """

    passed: bool
    bundle_version: str
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bundle validation
# ---------------------------------------------------------------------------

def validate_bundle(bundle_dir: str | Path) -> ValidationResult:
    """Validate a candidate bundle directory before promotion.

    Checks performed (rpi5.md §8.4):
    1. manifest.json loads and parses without error
    2. Required files are present (feature_contract.json)
    3. File checksums listed in the manifest all match
    4. Model artefact loads without error

    Parameters
    ----------
    bundle_dir:
        Path to the versioned bundle directory to validate.

    Returns
    -------
    ValidationResult
        Always returns — never raises.  Check ``result.passed``.
    """
    bundle_dir = Path(bundle_dir)
    errors: list[str] = []
    bundle_version = str(bundle_dir.name)  # fallback if manifest load fails

    # 1. Manifest integrity
    try:
        manifest = BundleManifest.load(bundle_dir)
        bundle_version = manifest.bundle_version
    except Exception as exc:
        errors.append(f"manifest load failed: {exc}")
        return ValidationResult(passed=False, bundle_version=bundle_version, errors=errors)

    # 2. Required artefact files
    required = ["feature_contract.json"]
    for fname in required:
        if not (bundle_dir / fname).exists():
            errors.append(f"required file missing: {fname}")

    # 3. Checksum verification
    checksum_errors = manifest.verify_checksums(bundle_dir)
    for fname, err_msg in checksum_errors.items():
        errors.append(f"checksum {fname}: {err_msg}")

    # 4. Model load
    try:
        _load_model_artifact(bundle_dir, manifest.model_type)
    except Exception as exc:
        errors.append(f"model load failed: {exc}")

    passed = len(errors) == 0
    if not passed:
        logger.warning(
            "Bundle validation failed for '%s': %s", bundle_version, errors
        )
    return ValidationResult(passed=passed, bundle_version=bundle_version, errors=errors)


# ---------------------------------------------------------------------------
# Pointer management (symlink-based, text-file fallback)
# ---------------------------------------------------------------------------

def _read_pointer(asset_dir: Path, pointer: str) -> str | None:
    """Return the bundle version name that *pointer* points to, or None."""
    ptr = asset_dir / pointer
    if not ptr.exists():
        return None
    if ptr.is_symlink():
        return ptr.readlink().name if hasattr(ptr.readlink(), "name") else str(ptr.readlink())
    if ptr.is_dir():
        return ptr.name
    return ptr.read_text(encoding="utf-8").strip() or None


def _write_pointer(asset_dir: Path, pointer: str, target_version: str) -> None:
    """Atomically repoint *pointer* to *target_version*.

    Uses symlink on POSIX (default), text-file on Windows or when symlink
    creation fails (e.g. CI with restricted permissions).
    """
    ptr = asset_dir / pointer

    # Remove existing pointer
    if ptr.is_symlink() or ptr.is_file():
        ptr.unlink()
    elif ptr.is_dir() and ptr.name == pointer:
        # Pointer is a real directory (development shortcut); cannot repoint it.
        # This is a no-op for the filesystem but the caller should be aware.
        logger.warning(
            "Cannot repoint '%s' — it is a real directory, not a symlink or text-file pointer. "
            "The filesystem pointer has NOT been updated to '%s'.",
            ptr, target_version,
        )
        return

    try:
        ptr.symlink_to(target_version)
    except (OSError, NotImplementedError):
        # Fallback: plain text file
        ptr.write_text(target_version, encoding="utf-8")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _upsert_active_bundle(
    db: ServingDB,
    manifest: BundleManifest,
    previous_version: str | None,
    promoted_at: datetime,
) -> None:
    """DELETE + INSERT into active_bundles for one asset (DuckDB 0.9 upsert)."""
    conn = db.connect()
    conn.execute(
        "DELETE FROM active_bundles WHERE asset_id = ?",
        [manifest.asset_id],
    )
    conn.execute(
        """
        INSERT INTO active_bundles
            (asset_id, source_ticker, release_id, bundle_version,
             model_type, feature_profile, promoted_at, previous_bundle_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            manifest.asset_id,
            manifest.source_ticker,
            manifest.release_id,
            manifest.bundle_version,
            manifest.model_type,
            manifest.feature_profile,
            promoted_at.isoformat(),
            previous_version,
        ],
    )


def _record_promotion_event(
    db: ServingDB,
    asset_id: str,
    release_id: str,
    bundle_version: str,
    status: PromotionStatus,
    promoted_at: datetime,
    previous_version: str | None = None,
    validation_errors: list[str] | None = None,
) -> None:
    """Insert one row into promotion_events."""
    import json

    errors_json: str | None = None
    if validation_errors:
        errors_json = json.dumps(validation_errors)

    conn = db.connect()
    conn.execute(
        """
        INSERT INTO promotion_events
            (event_id, asset_id, release_id, bundle_version,
             status, previous_version, validation_errors, promoted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            str(uuid.uuid4()),
            asset_id,
            release_id,
            bundle_version,
            status.value,
            previous_version,
            errors_json,
            promoted_at.isoformat(),
        ],
    )


# ---------------------------------------------------------------------------
# Per-asset promotion
# ---------------------------------------------------------------------------

def promote_asset_bundle(
    bundles_dir: str | Path,
    asset_id: str,
    new_bundle_version: str,
    db: ServingDB,
) -> ValidationResult:
    """Promote *new_bundle_version* to ``current`` for *asset_id*.

    Steps:
    1. Validate the candidate bundle
    2. On validation pass:
       - Repoint ``previous`` to the old ``current`` target
       - Repoint ``current`` to *new_bundle_version*
       - Upsert ``active_bundles`` in serving.duckdb
       - Record a ``promoted`` event
    3. On validation fail:
       - Leave ``current`` unchanged
       - Record a ``failed`` event

    Parameters
    ----------
    bundles_dir:
        Root of the bundle tree (``/srv/quant-risk/bundles`` on RPi5).
    asset_id:
        Semantic asset identifier.
    new_bundle_version:
        The bundle version string to promote (directory name under asset_id/).
    db:
        Open ServingDB connection.

    Returns
    -------
    ValidationResult
        Always returns — check ``result.passed`` for outcome.
    """
    bundles_dir = Path(bundles_dir)
    asset_dir = bundles_dir / asset_id
    candidate_dir = asset_dir / new_bundle_version
    promoted_at = datetime.now(timezone.utc)

    if not candidate_dir.is_dir():
        err = f"candidate bundle directory '{candidate_dir}' not found"
        logger.error("[%s] Promotion failed: %s", asset_id, err)
        _record_promotion_event(
            db, asset_id,
            release_id="unknown",
            bundle_version=new_bundle_version,
            status=PromotionStatus.failed,
            promoted_at=promoted_at,
            validation_errors=[err],
        )
        return ValidationResult(
            passed=False, bundle_version=new_bundle_version, errors=[err]
        )

    # 1. Validate
    result = validate_bundle(candidate_dir)

    # Load manifest for DB update (already loaded inside validate_bundle if passed)
    try:
        manifest = BundleManifest.load(candidate_dir)
    except Exception as exc:
        result.errors.append(f"manifest reload failed: {exc}")
        result.passed = False
        manifest = None

    if not result.passed or manifest is None:
        logger.warning(
            "[%s] Promotion validation failed — keeping current bundle. Errors: %s",
            asset_id, result.errors,
        )
        release_id = manifest.release_id if manifest else "unknown"
        _record_promotion_event(
            db, asset_id,
            release_id=release_id,
            bundle_version=new_bundle_version,
            status=PromotionStatus.failed,
            promoted_at=promoted_at,
            validation_errors=result.errors,
        )
        return result

    # 2. Repoint filesystem pointers
    previous_version = _read_pointer(asset_dir, "current")
    if previous_version:
        _write_pointer(asset_dir, "previous", previous_version)
        logger.debug("[%s] previous → %s", asset_id, previous_version)

    _write_pointer(asset_dir, "current", new_bundle_version)
    logger.info("[%s] current → %s", asset_id, new_bundle_version)

    # 3. Update serving.duckdb
    _upsert_active_bundle(db, manifest, previous_version, promoted_at)

    # 4. Record promotion event
    _record_promotion_event(
        db, asset_id,
        release_id=manifest.release_id,
        bundle_version=manifest.bundle_version,
        status=PromotionStatus.promoted,
        promoted_at=promoted_at,
        previous_version=previous_version,
    )

    logger.info(
        "[%s] Promoted to bundle_version=%s (previous=%s)",
        asset_id, new_bundle_version, previous_version,
    )
    return result


# ---------------------------------------------------------------------------
# Per-asset rollback
# ---------------------------------------------------------------------------

def rollback_asset_bundle(
    bundles_dir: str | Path,
    asset_id: str,
    db: ServingDB,
) -> bool:
    """Roll back *asset_id* to the ``previous`` bundle.

    Steps:
    1. Read ``previous`` pointer target
    2. Repoint ``current`` to the previous bundle version
    3. Update ``active_bundles`` in serving.duckdb
    4. Record a ``promoted`` event for the rollback

    Parameters
    ----------
    bundles_dir:
        Root of the bundle tree.
    asset_id:
        Semantic asset identifier.
    db:
        Open ServingDB connection.

    Returns
    -------
    bool
        True if rollback succeeded, False if no ``previous`` bundle exists.
    """
    bundles_dir = Path(bundles_dir)
    asset_dir = bundles_dir / asset_id
    rolled_back_at = datetime.now(timezone.utc)

    previous_version = _read_pointer(asset_dir, "previous")
    if not previous_version:
        logger.warning("[%s] Rollback requested but no previous bundle exists.", asset_id)
        return False

    previous_dir = asset_dir / previous_version
    if not previous_dir.is_dir():
        logger.error(
            "[%s] Rollback failed: previous bundle directory '%s' not found.",
            asset_id, previous_dir,
        )
        return False

    # Load manifest for DB update
    try:
        manifest = BundleManifest.load(previous_dir)
    except Exception as exc:
        logger.error("[%s] Rollback failed: manifest load error: %s", asset_id, exc)
        return False

    old_current = _read_pointer(asset_dir, "current")

    # Repoint current → previous target
    _write_pointer(asset_dir, "current", previous_version)
    logger.info("[%s] Rolled back current → %s", asset_id, previous_version)

    # Update serving.duckdb
    _upsert_active_bundle(db, manifest, old_current, rolled_back_at)

    # Record rollback as a promotion event
    _record_promotion_event(
        db, asset_id,
        release_id=manifest.release_id,
        bundle_version=manifest.bundle_version,
        status=PromotionStatus.promoted,
        promoted_at=rolled_back_at,
        previous_version=old_current,
    )

    logger.info(
        "[%s] Rollback complete — active bundle is now %s", asset_id, previous_version
    )
    return True
