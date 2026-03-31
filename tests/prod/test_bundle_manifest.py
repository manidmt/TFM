'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Tests for the production bundle manifest (bundle_manifest.py).

Covers:
- BundleManifest construction with all required fields.
- Field validators: release_id format, bundle_version format, horizon_days.
- Cross-field validator: bundle_version date must match release_id date.
- Serialise/deserialise round-trip (to_json / from_json).
- save() / load() round-trip using a tmp_path directory.
- verify_checksums: pass and fail paths.
- make_manifest() factory fills python_version and dependency_versions defaults.
- DependencyVersions allows extra fields.
'''

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quant_risk.prod.bundle_manifest import (
    BundleManifest,
    DependencyVersions,
    FileChecksum,
    make_manifest,
)


# ---------------------------------------------------------------------------
# Minimal valid manifest fixture
# ---------------------------------------------------------------------------

def _minimal_manifest(**overrides) -> BundleManifest:
    defaults = dict(
        release_id="prod_2026-03-27",
        bundle_version="2026-03-27_prod_ab12cd3",
        asset_id="us_equities",
        source_ticker="^GSPC",
        horizon_days=5,
        model_type="xgb",
        feature_profile="gkg_v1_features_compact",
        training_run_ref="abc123def456",
        created_at=datetime(2026, 3, 27, 10, 0, 0, tzinfo=timezone.utc),
        python_version="3.11.14",
        dependency_versions=DependencyVersions(python="3.11.14", xgboost="2.1.4"),
    )
    defaults.update(overrides)
    return BundleManifest(**defaults)


# ---------------------------------------------------------------------------
# Happy-path construction
# ---------------------------------------------------------------------------

def test_minimal_manifest_constructs():
    m = _minimal_manifest()
    assert m.asset_id == "us_equities"
    assert m.horizon_days == 5
    assert m.model_type == "xgb"


def test_manifest_allows_optional_fields():
    m = _minimal_manifest(
        chain_variants_config="src/quant_risk/models/econometric/chain_variants.yaml",
        tabular_hyperparams={"max_depth": 5, "learning_rate": 0.05},
        notes="Champion from sweep 2026-03-27",
    )
    assert m.chain_variants_config is not None
    assert m.tabular_hyperparams["max_depth"] == 5


def test_manifest_horizon_20_accepted():
    m = _minimal_manifest(horizon_days=20)
    assert m.horizon_days == 20


# ---------------------------------------------------------------------------
# Field validators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_release_id", [
    "2026-03-27",          # missing "prod_" prefix
    "prod_27-03-2026",     # wrong date order
    "prod_2026/03/27",     # slashes instead of dashes
    "PROD_2026-03-27",     # uppercase
    "",
])
def test_invalid_release_id_raises(bad_release_id):
    with pytest.raises(Exception):
        _minimal_manifest(release_id=bad_release_id)


@pytest.mark.parametrize("bad_bundle_version", [
    "2026-03-27_abc",               # no "prod_" in middle
    "prod_2026-03-27_ab12cd3",      # date and prod_ swapped
    "2026-03-27_prod_",             # empty shortref
    "2026-03-27_prod_ABCDEFG",      # uppercase shortref
    "",
])
def test_invalid_bundle_version_raises(bad_bundle_version):
    with pytest.raises(Exception):
        _minimal_manifest(bundle_version=bad_bundle_version)


@pytest.mark.parametrize("bad_horizon", [1, 10, 30, 0, -5])
def test_invalid_horizon_raises(bad_horizon):
    with pytest.raises(Exception):
        _minimal_manifest(horizon_days=bad_horizon)


def test_bundle_version_date_mismatch_with_release_id_raises():
    # bundle_version starts with 2026-03-28 but release_id is prod_2026-03-27
    with pytest.raises(Exception, match="must start with the release date"):
        _minimal_manifest(
            release_id="prod_2026-03-27",
            bundle_version="2026-03-28_prod_ab12cd3",
        )


# ---------------------------------------------------------------------------
# FileChecksum validation
# ---------------------------------------------------------------------------

def test_file_checksum_valid():
    fc = FileChecksum(filename="model/model.pkl", sha256="a" * 64)
    assert fc.sha256 == "a" * 64


def test_file_checksum_uppercase_normalised():
    fc = FileChecksum(filename="model/model.pkl", sha256="A" * 64)
    assert fc.sha256 == "a" * 64


def test_file_checksum_wrong_length_raises():
    with pytest.raises(Exception):
        FileChecksum(filename="model/model.pkl", sha256="abcdef")


# ---------------------------------------------------------------------------
# DependencyVersions
# ---------------------------------------------------------------------------

def test_dependency_versions_allows_extra_fields():
    dv = DependencyVersions(python="3.11", arch="7.0.0", hmmlearn="0.3.0")
    assert dv.arch == "7.0.0"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

def test_json_round_trip():
    m = _minimal_manifest()
    reconstructed = BundleManifest.from_json(m.to_json())
    assert reconstructed.release_id == m.release_id
    assert reconstructed.bundle_version == m.bundle_version
    assert reconstructed.asset_id == m.asset_id
    assert reconstructed.created_at == m.created_at


def test_json_contains_expected_keys():
    m = _minimal_manifest()
    data = json.loads(m.to_json())
    required_keys = {
        "release_id", "bundle_version", "asset_id", "source_ticker",
        "horizon_days", "model_type", "feature_profile", "training_run_ref",
        "created_at", "python_version", "dependency_versions",
    }
    assert required_keys.issubset(data.keys())


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path: Path):
    m = _minimal_manifest()
    saved_path = m.save(tmp_path)
    assert saved_path.name == "manifest.json"
    loaded = BundleManifest.load(tmp_path)
    assert loaded.bundle_version == m.bundle_version
    assert loaded.dependency_versions.python == m.dependency_versions.python


def test_load_raises_if_no_manifest_json(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        BundleManifest.load(tmp_path)


def test_save_raises_if_dir_does_not_exist(tmp_path: Path):
    m = _minimal_manifest()
    with pytest.raises(NotADirectoryError):
        m.save(tmp_path / "nonexistent_dir")


# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------

def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_verify_checksums_passes_when_correct(tmp_path: Path):
    content = b"fake model weights"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_file = model_dir / "model.pkl"
    model_file.write_bytes(content)

    m = _minimal_manifest(
        file_checksums=[
            FileChecksum(filename="model/model.pkl", sha256=_sha256(content))
        ]
    )
    errors = m.verify_checksums(tmp_path)
    assert errors == {}


def test_verify_checksums_detects_mismatch(tmp_path: Path):
    content = b"correct content"
    wrong_hash = _sha256(b"wrong content")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pkl").write_bytes(content)

    m = _minimal_manifest(
        file_checksums=[
            FileChecksum(filename="model/model.pkl", sha256=wrong_hash)
        ]
    )
    errors = m.verify_checksums(tmp_path)
    assert "model/model.pkl" in errors
    assert "mismatch" in errors["model/model.pkl"]


def test_verify_checksums_reports_missing_file(tmp_path: Path):
    m = _minimal_manifest(
        file_checksums=[
            FileChecksum(filename="model/missing.pkl", sha256="a" * 64)
        ]
    )
    errors = m.verify_checksums(tmp_path)
    assert "model/missing.pkl" in errors
    assert "not found" in errors["model/missing.pkl"]


def test_verify_checksums_empty_list_always_passes(tmp_path: Path):
    m = _minimal_manifest(file_checksums=[])
    assert m.verify_checksums(tmp_path) == {}


# ---------------------------------------------------------------------------
# make_manifest factory
# ---------------------------------------------------------------------------

def test_make_manifest_fills_python_version():
    import sys
    m = make_manifest(
        release_id="prod_2026-03-27",
        bundle_version="2026-03-27_prod_ab12cd3",
        asset_id="gold",
        source_ticker="GLD",
        horizon_days=5,
        model_type="xgb",
        feature_profile="gkg_v1_features_compact",
        training_run_ref="run_xyz",
    )
    expected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    assert m.python_version == expected


def test_make_manifest_created_at_is_utc():
    m = make_manifest(
        release_id="prod_2026-03-27",
        bundle_version="2026-03-27_prod_ab12cd3",
        asset_id="gold",
        source_ticker="GLD",
        horizon_days=5,
        model_type="tabpfn",
        feature_profile="gkg_v1_features_compact",
        training_run_ref="run_xyz",
    )
    assert m.created_at.tzinfo is not None
