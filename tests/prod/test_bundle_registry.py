"""
Tests for src/quant_risk/prod/bundle_registry.py and src/quant_risk/prod/promotion.py.

Coverage (bundle_registry):
- get_bundle_dir: symlink, text-file, dir conventions; missing pointer; missing asset
- get_active_bundle_dir / get_previous_bundle_dir
- has_active_bundle
- list_bundle_versions: excludes pointers, sorts alphabetically
- load_active_manifest

Coverage (promotion):
- validate_bundle: pass, missing manifest, missing feature_contract, checksum failure, model load failure
- promote_asset_bundle: happy path, validation fail, missing candidate dir
- rollback_asset_bundle: happy path, no previous, missing previous dir
- pointer management: current/previous correctly written after promote/rollback
- serving.duckdb: active_bundles and promotion_events updated
"""

from __future__ import annotations

import json
import os
import pickle
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quant_risk.prod.bundle_manifest import make_manifest
from quant_risk.prod.bundle_registry import (
    get_active_bundle_dir,
    get_bundle_dir,
    get_previous_bundle_dir,
    has_active_bundle,
    list_bundle_versions,
    load_active_manifest,
)
from quant_risk.prod.promotion import (
    ValidationResult,
    promote_asset_bundle,
    rollback_asset_bundle,
    validate_bundle,
)
from quant_risk.prod.schemas import PromotionStatus
from quant_risk.prod.serving.duckdb import ServingDB


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "serving.duckdb")
    sdb = ServingDB(path)
    sdb.create_schema()
    yield sdb
    sdb.close()


class _StubModel:
    def predict_proba(self, X):
        import numpy as np
        return [[0.2, 0.3, 0.5]] * len(X)


def _make_bundle_dir(
    asset_dir: Path,
    bundle_version: str = "2026-03-27_prod_abc1234",
    model_ok: bool = True,
    include_feature_contract: bool = True,
    checksum_ok: bool = True,
    release_id: str | None = None,
) -> Path:
    """Create a minimal bundle directory under asset_dir."""
    bundle_dir = asset_dir / bundle_version
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model_dir = bundle_dir / "model"
    model_dir.mkdir()

    # model/model.pkl
    if model_ok:
        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(_StubModel(), f)
    # else: leave model/ empty so load fails

    # feature_contract.json
    if include_feature_contract:
        (bundle_dir / "feature_contract.json").write_text(
            json.dumps(["logret", "rv_5", "rv_20"])
        )

    # inference_config.json
    (bundle_dir / "inference_config.json").write_text(
        json.dumps({"class_labels": ["low", "medium", "high"]})
    )

    # calibration.json
    (bundle_dir / "calibration.json").write_text(json.dumps({"method": "none"}))

    # manifest.json — with optional checksum for feature_contract
    file_checksums = []
    if include_feature_contract and not checksum_ok:
        file_checksums = [{"filename": "feature_contract.json", "sha256": "a" * 64}]
    elif include_feature_contract and checksum_ok:
        # Compute real checksum so validation passes
        import hashlib
        h = hashlib.sha256((bundle_dir / "feature_contract.json").read_bytes()).hexdigest()
        file_checksums = [{"filename": "feature_contract.json", "sha256": h}]

    # Derive release_id from bundle_version date prefix if not provided
    if release_id is None:
        date_prefix = bundle_version.split("_prod_")[0]  # "2026-03-27"
        release_id = f"prod_{date_prefix}"

    manifest = make_manifest(
        release_id=release_id,
        bundle_version=bundle_version,
        asset_id=asset_dir.name,
        source_ticker="^GSPC",
        horizon_days=5,
        model_type="xgb",
        feature_profile="gkg_v1_features_compact",
        training_run_ref="mlflow_abc123",
        file_checksums=file_checksums,
    )
    manifest.save(bundle_dir)

    return bundle_dir


def _set_pointer(asset_dir: Path, pointer: str, version: str) -> None:
    """Create a symlink or text-file pointer."""
    ptr = asset_dir / pointer
    if ptr.is_symlink() or ptr.is_file():
        ptr.unlink()
    try:
        ptr.symlink_to(version)
    except (OSError, NotImplementedError):
        ptr.write_text(version)


# ---------------------------------------------------------------------------
# bundle_registry — get_bundle_dir
# ---------------------------------------------------------------------------

class TestGetBundleDir:
    def test_symlink_convention(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir)
        _set_pointer(asset_dir, "current", "2026-03-27_prod_abc1234")
        result = get_bundle_dir(tmp_path, "us_equities", "current")
        assert result.is_dir()
        assert result.name == "2026-03-27_prod_abc1234"

    def test_textfile_convention(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir)
        ptr = asset_dir / "current"
        ptr.write_text("2026-03-27_prod_abc1234")
        result = get_bundle_dir(tmp_path, "us_equities", "current")
        assert result.is_dir()

    def test_missing_asset_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No bundle directory"):
            get_bundle_dir(tmp_path, "nonexistent", "current")

    def test_missing_pointer_raises(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="pointer 'current'"):
            get_bundle_dir(tmp_path, "us_equities", "current")


# ---------------------------------------------------------------------------
# bundle_registry — get_active / get_previous / has_active
# ---------------------------------------------------------------------------

class TestActivePreviousHelpers:
    def test_get_active_bundle_dir(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir)
        _set_pointer(asset_dir, "current", "2026-03-27_prod_abc1234")
        result = get_active_bundle_dir(tmp_path, "us_equities")
        assert result.is_dir()

    def test_get_previous_none_when_absent(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir)
        _set_pointer(asset_dir, "current", "2026-03-27_prod_abc1234")
        assert get_previous_bundle_dir(tmp_path, "us_equities") is None

    def test_get_previous_returns_dir(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-20_prod_f45e9aa")
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        _set_pointer(asset_dir, "current", "2026-03-27_prod_abc1234")
        _set_pointer(asset_dir, "previous", "2026-03-20_prod_f45e9aa")
        result = get_previous_bundle_dir(tmp_path, "us_equities")
        assert result is not None
        assert result.name == "2026-03-20_prod_f45e9aa"

    def test_has_active_bundle_true(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir)
        _set_pointer(asset_dir, "current", "2026-03-27_prod_abc1234")
        assert has_active_bundle(tmp_path, "us_equities") is True

    def test_has_active_bundle_false(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        assert has_active_bundle(tmp_path, "us_equities") is False

    def test_has_active_bundle_missing_asset(self, tmp_path):
        assert has_active_bundle(tmp_path, "nonexistent") is False


# ---------------------------------------------------------------------------
# bundle_registry — list_bundle_versions
# ---------------------------------------------------------------------------

class TestListBundleVersions:
    def test_empty_when_no_asset_dir(self, tmp_path):
        assert list_bundle_versions(tmp_path, "us_equities") == []

    def test_excludes_current_previous(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        _make_bundle_dir(asset_dir, "2026-03-20_prod_f45e9aa")
        _set_pointer(asset_dir, "current", "2026-03-27_prod_abc1234")
        _set_pointer(asset_dir, "previous", "2026-03-20_prod_f45e9aa")
        versions = list_bundle_versions(tmp_path, "us_equities")
        assert "current" not in versions
        assert "previous" not in versions

    def test_sorted_alphabetically(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        vnames = ["2026-03-27_prod_abc1234", "2026-03-20_prod_f45e9aa", "2026-03-13_prod_deadb0b"]
        for v in vnames:
            _make_bundle_dir(asset_dir, v)
        versions = list_bundle_versions(tmp_path, "us_equities")
        assert versions == sorted(vnames)

    def test_returns_version_names(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        versions = list_bundle_versions(tmp_path, "us_equities")
        assert versions == ["2026-03-27_prod_abc1234"]


# ---------------------------------------------------------------------------
# bundle_registry — load_active_manifest
# ---------------------------------------------------------------------------

class TestLoadActiveManifest:
    def test_loads_manifest(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        _set_pointer(asset_dir, "current", "2026-03-27_prod_abc1234")
        manifest = load_active_manifest(tmp_path, "us_equities")
        assert manifest.bundle_version == "2026-03-27_prod_abc1234"
        assert manifest.asset_id == "us_equities"


# ---------------------------------------------------------------------------
# validate_bundle
# ---------------------------------------------------------------------------

class TestValidateBundle:
    def test_valid_bundle_passes(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        bundle_dir = _make_bundle_dir(asset_dir)
        result = validate_bundle(bundle_dir)
        assert result.passed is True
        assert result.errors == []

    def test_missing_manifest_fails(self, tmp_path):
        bundle_dir = tmp_path / "2026-03-27_prod_abc1234"
        bundle_dir.mkdir()
        (bundle_dir / "model").mkdir()
        result = validate_bundle(bundle_dir)
        assert result.passed is False
        assert any("manifest" in e for e in result.errors)

    def test_missing_feature_contract_fails(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        bundle_dir = _make_bundle_dir(asset_dir, include_feature_contract=False)
        result = validate_bundle(bundle_dir)
        assert result.passed is False
        assert any("feature_contract" in e for e in result.errors)

    def test_bad_checksum_fails(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        bundle_dir = _make_bundle_dir(asset_dir, checksum_ok=False)
        result = validate_bundle(bundle_dir)
        assert result.passed is False
        assert any("checksum" in e for e in result.errors)

    def test_missing_model_fails(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        bundle_dir = _make_bundle_dir(asset_dir, model_ok=False)
        result = validate_bundle(bundle_dir)
        assert result.passed is False
        assert any("model" in e for e in result.errors)

    def test_bundle_version_in_result(self, tmp_path):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        bundle_dir = _make_bundle_dir(asset_dir)
        result = validate_bundle(bundle_dir)
        assert result.bundle_version == "2026-03-27_prod_abc1234"


# ---------------------------------------------------------------------------
# promote_asset_bundle
# ---------------------------------------------------------------------------

class TestPromoteAssetBundle:
    def test_happy_path_passes(self, tmp_path, db):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        result = promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_abc1234", db)
        assert result.passed is True

    def test_current_pointer_written(self, tmp_path, db):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_abc1234", db)
        current_ptr = asset_dir / "current"
        assert current_ptr.exists() or current_ptr.is_symlink()

    def test_active_bundles_updated(self, tmp_path, db):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_abc1234", db)
        rows = db.connect().execute(
            "SELECT bundle_version FROM active_bundles WHERE asset_id='us_equities'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "2026-03-27_prod_abc1234"

    def test_promotion_event_recorded(self, tmp_path, db):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_abc1234", db)
        rows = db.connect().execute(
            "SELECT status FROM promotion_events WHERE asset_id='us_equities'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "promoted"

    def test_second_promotion_moves_to_previous(self, tmp_path, db):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-20_prod_f45e9aa",
                         release_id="prod_2026-03-20")
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234",
                         release_id="prod_2026-03-27")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-20_prod_f45e9aa", db)
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_abc1234", db)
        prev = get_previous_bundle_dir(tmp_path, "us_equities")
        assert prev is not None
        assert prev.name == "2026-03-20_prod_f45e9aa"

    def test_validation_fail_keeps_current(self, tmp_path, db):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        # First promote a valid bundle
        _make_bundle_dir(asset_dir, "2026-03-20_prod_f45e9aa",
                         release_id="prod_2026-03-20")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-20_prod_f45e9aa", db)
        # Now try to promote an invalid bundle (no model)
        _make_bundle_dir(asset_dir, "2026-03-27_prod_bad0000",
                         model_ok=False, release_id="prod_2026-03-27")
        result = promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_bad0000", db)
        assert result.passed is False
        # current should still point to the valid bundle
        current_dir = get_active_bundle_dir(tmp_path, "us_equities")
        assert current_dir.name == "2026-03-20_prod_f45e9aa"

    def test_validation_fail_records_failed_event(self, tmp_path, db):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_bad0000",
                         model_ok=False, release_id="prod_2026-03-27")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_bad0000", db)
        rows = db.connect().execute(
            "SELECT status FROM promotion_events WHERE asset_id='us_equities'"
        ).fetchall()
        assert any(r[0] == "failed" for r in rows)

    def test_missing_candidate_dir_fails(self, tmp_path, db):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        result = promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_nonexist", db)
        assert result.passed is False
        assert any("not found" in e for e in result.errors)

    def test_promote_idempotent(self, tmp_path, db):
        """Promoting the same version twice should succeed and leave active_bundles with one row."""
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_abc1234", db)
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_abc1234", db)
        count = db.connect().execute(
            "SELECT COUNT(*) FROM active_bundles WHERE asset_id='us_equities'"
        ).fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# rollback_asset_bundle
# ---------------------------------------------------------------------------

class TestRollbackAssetBundle:
    def test_rollback_happy_path(self, tmp_path, db):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-20_prod_f45e9aa",
                         release_id="prod_2026-03-20")
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234",
                         release_id="prod_2026-03-27")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-20_prod_f45e9aa", db)
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_abc1234", db)
        # current = abc1234, previous = f45e9aa
        success = rollback_asset_bundle(tmp_path, "us_equities", db)
        assert success is True

    def test_rollback_restores_current(self, tmp_path, db):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-20_prod_f45e9aa",
                         release_id="prod_2026-03-20")
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234",
                         release_id="prod_2026-03-27")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-20_prod_f45e9aa", db)
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_abc1234", db)
        rollback_asset_bundle(tmp_path, "us_equities", db)
        current_dir = get_active_bundle_dir(tmp_path, "us_equities")
        assert current_dir.name == "2026-03-20_prod_f45e9aa"

    def test_rollback_updates_active_bundles(self, tmp_path, db):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-20_prod_f45e9aa",
                         release_id="prod_2026-03-20")
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234",
                         release_id="prod_2026-03-27")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-20_prod_f45e9aa", db)
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_abc1234", db)
        rollback_asset_bundle(tmp_path, "us_equities", db)
        row = db.connect().execute(
            "SELECT bundle_version FROM active_bundles WHERE asset_id='us_equities'"
        ).fetchone()
        assert row[0] == "2026-03-20_prod_f45e9aa"

    def test_rollback_no_previous_returns_false(self, tmp_path, db):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_abc1234", db)
        # No previous exists (first-ever promotion)
        success = rollback_asset_bundle(tmp_path, "us_equities", db)
        assert success is False

    def test_rollback_records_promotion_event(self, tmp_path, db):
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-20_prod_f45e9aa",
                         release_id="prod_2026-03-20")
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234",
                         release_id="prod_2026-03-27")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-20_prod_f45e9aa", db)
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-27_prod_abc1234", db)
        rollback_asset_bundle(tmp_path, "us_equities", db)
        # Should have 3 events: promote f45e9aa, promote abc1234, rollback to f45e9aa
        count = db.connect().execute(
            "SELECT COUNT(*) FROM promotion_events WHERE asset_id='us_equities'"
        ).fetchone()[0]
        assert count == 3


# ---------------------------------------------------------------------------
# _write_pointer — directory-as-pointer edge case
# ---------------------------------------------------------------------------

class TestWritePointerDirectoryCase:
    """Tests for the development-shortcut case where 'current' is a real directory.

    When the 'current' pointer is a real directory (not a symlink or text file),
    _write_pointer should log a warning and return without touching the filesystem.
    The DB update in promote_asset_bundle still proceeds before _write_pointer is
    called, so active_bundles and promotion_events are written regardless.
    """

    def test_directory_pointer_filesystem_not_updated(self, tmp_path, db):
        """If 'current' is a real directory, the pointer is NOT rewritten."""
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        # Create 'current' as a real directory (dev shortcut) pointing to the bundle
        current_dir = asset_dir / "current"
        current_dir.mkdir()
        # Copy bundle files into the directory so bundle_registry can resolve it
        import shutil
        for f in (asset_dir / "2026-03-27_prod_abc1234").iterdir():
            if f.is_file():
                shutil.copy(f, current_dir / f.name)

        # Attempt to promote new bundle — _write_pointer will hit the directory branch
        _make_bundle_dir(asset_dir, "2026-03-28_prod_def5678")
        result = promote_asset_bundle(tmp_path, "us_equities", "2026-03-28_prod_def5678", db)

        # Promotion validation passes (new bundle is valid)
        assert result.passed is True

        # The 'current' pointer (as a real directory) was NOT rewritten
        assert (asset_dir / "current").is_dir()
        assert not (asset_dir / "current").is_symlink()

    def test_directory_pointer_db_still_updated(self, tmp_path, db):
        """DB (active_bundles) is updated even when filesystem pointer is a directory."""
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        current_dir = asset_dir / "current"
        current_dir.mkdir()
        import shutil
        for f in (asset_dir / "2026-03-27_prod_abc1234").iterdir():
            if f.is_file():
                shutil.copy(f, current_dir / f.name)

        _make_bundle_dir(asset_dir, "2026-03-28_prod_def5678")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-28_prod_def5678", db)

        row = db.connect().execute(
            "SELECT bundle_version FROM active_bundles WHERE asset_id='us_equities'"
        ).fetchone()
        assert row is not None
        assert row[0] == "2026-03-28_prod_def5678"

    def test_directory_pointer_promotion_event_recorded(self, tmp_path, db):
        """Promotion event is recorded even when filesystem pointer is a directory."""
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        current_dir = asset_dir / "current"
        current_dir.mkdir()
        import shutil
        for f in (asset_dir / "2026-03-27_prod_abc1234").iterdir():
            if f.is_file():
                shutil.copy(f, current_dir / f.name)

        _make_bundle_dir(asset_dir, "2026-03-28_prod_def5678")
        promote_asset_bundle(tmp_path, "us_equities", "2026-03-28_prod_def5678", db)

        count = db.connect().execute(
            "SELECT COUNT(*) FROM promotion_events "
            "WHERE asset_id='us_equities' AND status='promoted'"
        ).fetchone()[0]
        assert count == 1

    def test_directory_pointer_logs_warning(self, tmp_path, db, caplog):
        """_write_pointer emits a warning when pointer is a real directory."""
        import logging
        asset_dir = tmp_path / "us_equities"
        asset_dir.mkdir()
        _make_bundle_dir(asset_dir, "2026-03-27_prod_abc1234")
        current_dir = asset_dir / "current"
        current_dir.mkdir()
        import shutil
        for f in (asset_dir / "2026-03-27_prod_abc1234").iterdir():
            if f.is_file():
                shutil.copy(f, current_dir / f.name)

        _make_bundle_dir(asset_dir, "2026-03-28_prod_def5678")
        with caplog.at_level(logging.WARNING, logger="quant_risk.prod.promotion"):
            promote_asset_bundle(tmp_path, "us_equities", "2026-03-28_prod_def5678", db)

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Cannot repoint" in m for m in warning_msgs), (
            f"Expected 'Cannot repoint' warning, got: {warning_msgs}"
        )
