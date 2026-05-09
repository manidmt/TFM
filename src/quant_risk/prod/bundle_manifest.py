'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Bundle manifest schema and I/O helpers.

A bundle is a versioned directory containing a trained model artifact plus all
the metadata needed to reproduce and validate inference on the RPi5.
The manifest.json file inside each bundle is the authoritative contract that
links a training run on the research machine to production inference.

Bundle directory structure (rpi5.md §8.1):

    bundles/
      <asset_id>/
        current  -> <bundle_version>/    (symlink)
        previous -> <bundle_version>/    (symlink)
        <bundle_version>/
          manifest.json
          model/
            ...
          feature_contract.json
          inference_config.json
          calibration.json
          thresholds.json

Naming conventions (rpi5.md §8.1, §8.3):
  release_id:     prod_<YYYY-MM-DD>          e.g. "prod_2026-03-27"
  bundle_version: <YYYY-MM-DD>_<release_id>_<short_ref>
                                             e.g. "2026-03-27_prod_ab12cd3"

Reference: rpi5.md §8
'''

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Naming-convention helpers
# ---------------------------------------------------------------------------

_RELEASE_ID_RE = re.compile(r"^prod_\d{4}-\d{2}-\d{2}$")
_BUNDLE_VERSION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_prod_[0-9a-f]{7,}$")


def _validate_release_id(v: str) -> str:
    if not _RELEASE_ID_RE.match(v):
        raise ValueError(
            f"release_id '{v}' does not match expected format 'prod_YYYY-MM-DD'."
        )
    return v


def _validate_bundle_version(v: str) -> str:
    if not _BUNDLE_VERSION_RE.match(v):
        raise ValueError(
            f"bundle_version '{v}' does not match expected format "
            "'YYYY-MM-DD_prod_<shortref>'."
        )
    return v


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class FileChecksum(BaseModel):
    """SHA-256 checksum for a single file inside the bundle."""

    filename: str = Field(..., description="Path relative to the bundle root.")
    sha256: str = Field(..., description="Lowercase hex SHA-256 digest.")

    @field_validator("sha256")
    @classmethod
    def _hex_format(cls, v: str) -> str:
        v = v.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError(f"sha256 must be a 64-character hex string, got: '{v}'")
        return v


class DependencyVersions(BaseModel):
    """Key library versions at training time (used for compatibility checks on RPi5)."""

    python: str
    numpy: str | None = None
    pandas: str | None = None
    scikit_learn: str | None = None
    xgboost: str | None = None
    tabpfn: str | None = None
    torch: str | None = None

    model_config = {"extra": "allow"}  # allow additional libraries


# ---------------------------------------------------------------------------
# BundleManifest
# ---------------------------------------------------------------------------

class BundleManifest(BaseModel):
    """Complete bundle manifest stored as manifest.json inside each bundle directory.

    All fields required at promotion time.  Optional fields are allowed during
    development and must be populated before a bundle can be promoted to
    ``current`` on the RPi5.

    Reference: rpi5.md §8.2
    """

    # Identity
    release_id: str = Field(..., description="Release identifier, e.g. 'prod_2026-03-27'.")
    bundle_version: str = Field(
        ..., description="Bundle version slug, e.g. '2026-03-27_prod_ab12cd3'."
    )
    asset_id: str = Field(..., description="Semantic asset identifier from config/prod/assets.yaml.")
    source_ticker: str = Field(..., description="Raw ticker symbol, e.g. '^GSPC'.")

    # Model
    horizon_days: int = Field(..., ge=1, description="Forecast horizon in calendar days.")
    model_type: str = Field(..., description="Model type identifier, e.g. 'xgb' or 'tabpfn'.")
    feature_profile: str = Field(
        ..., description="Feature profile name as used in config/features.yaml."
    )

    # Provenance
    training_run_ref: str = Field(
        ..., description="MLflow run ID or equivalent training provenance reference."
    )
    created_at: datetime = Field(
        ..., description="UTC timestamp at which the bundle was packaged."
    )
    python_version: str = Field(..., description="Python version string, e.g. '3.11.14'.")
    dependency_versions: DependencyVersions = Field(
        ..., description="Key library versions at training time."
    )

    # Integrity
    file_checksums: list[FileChecksum] = Field(
        default_factory=list,
        description="SHA-256 checksums for all files inside the bundle.",
    )

    # Optional enrichment
    chain_variants_config: str | None = Field(
        None, description="Path to the chain_variants yaml used for the econometric chain."
    )
    tabular_hyperparams: dict[str, Any] = Field(
        default_factory=dict,
        description="Tabular model hyperparameters logged at training time.",
    )
    notes: str | None = Field(None, description="Free-text notes about this bundle.")

    # Validators
    @field_validator("release_id")
    @classmethod
    def _check_release_id(cls, v: str) -> str:
        return _validate_release_id(v)

    @field_validator("bundle_version")
    @classmethod
    def _check_bundle_version(cls, v: str) -> str:
        return _validate_bundle_version(v)

    @field_validator("horizon_days")
    @classmethod
    def _check_horizon(cls, v: int) -> int:
        if v not in (5, 20):
            raise ValueError(f"horizon_days must be 5 or 20, got {v}.")
        return v

    @model_validator(mode="after")
    def _bundle_version_matches_release(self) -> "BundleManifest":
        # bundle_version must embed the same date as release_id
        release_date = self.release_id.replace("prod_", "")  # "2026-03-27"
        if not self.bundle_version.startswith(release_date):
            raise ValueError(
                f"bundle_version '{self.bundle_version}' must start with the "
                f"release date '{release_date}' derived from release_id '{self.release_id}'."
            )
        return self

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string (UTC datetimes as ISO-8601)."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "BundleManifest":
        """Deserialise from JSON string."""
        return cls.model_validate_json(text)

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "BundleManifest":
        """Load and validate manifest.json from *bundle_dir*."""
        path = Path(bundle_dir) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"manifest.json not found in '{bundle_dir}'.")
        return cls.from_json(path.read_text(encoding="utf-8"))

    def save(self, bundle_dir: str | Path) -> Path:
        """Write manifest.json into *bundle_dir* (directory must exist)."""
        bundle_dir = Path(bundle_dir)
        if not bundle_dir.is_dir():
            raise NotADirectoryError(f"Bundle directory does not exist: '{bundle_dir}'.")
        path = bundle_dir / "manifest.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Checksum helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sha256(path: str | Path) -> str:
        """Return the lowercase hex SHA-256 digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify_checksums(self, bundle_dir: str | Path) -> dict[str, str]:
        """Verify all listed file checksums against the actual files.

        Returns
        -------
        dict[str, str]
            Mapping of filename -> error message for any mismatches.
            Empty dict means all checksums passed.
        """
        bundle_dir = Path(bundle_dir)
        errors: dict[str, str] = {}
        for entry in self.file_checksums:
            fpath = bundle_dir / entry.filename
            if not fpath.exists():
                errors[entry.filename] = "file not found"
                continue
            actual = self.compute_sha256(fpath)
            if actual != entry.sha256:
                errors[entry.filename] = f"checksum mismatch (expected {entry.sha256}, got {actual})"
        return errors


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_manifest(
    *,
    release_id: str,
    bundle_version: str,
    asset_id: str,
    source_ticker: str,
    horizon_days: int,
    model_type: str,
    feature_profile: str,
    training_run_ref: str,
    python_version: str | None = None,
    dependency_versions: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BundleManifest:
    """Convenience factory that fills in defaults for non-critical fields."""
    if python_version is None:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if dependency_versions is None:
        dependency_versions = {"python": python_version}
    elif "python" not in dependency_versions:
        dependency_versions = {"python": python_version, **dependency_versions}

    return BundleManifest(
        release_id=release_id,
        bundle_version=bundle_version,
        asset_id=asset_id,
        source_ticker=source_ticker,
        horizon_days=horizon_days,
        model_type=model_type,
        feature_profile=feature_profile,
        training_run_ref=training_run_ref,
        created_at=datetime.now(timezone.utc),
        python_version=python_version,
        dependency_versions=DependencyVersions(**dependency_versions),
        **kwargs,
    )
