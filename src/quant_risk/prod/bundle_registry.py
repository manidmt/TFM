'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Bundle directory resolution and registry helpers.

The bundle registry is the filesystem tree at `bundles/`.  Each asset has its
own sub-directory containing versioned bundle directories plus two pointer
files (`current` and `previous`).

Layout (rpi5.md §8.1):

    bundles/
      <asset_id>/
        current   -> <bundle_version>/    (symlink)
        previous  -> <bundle_version>/    (symlink, may be absent)
        <bundle_version>/
          manifest.json
          model/
          feature_contract.json
          inference_config.json
          calibration.json
          thresholds.json

This module provides path helpers and light query utilities.  The actual
promotion logic (pointer rewrites + DB updates) lives in promotion.py.

Reference: rpi5.md §8, §9
'''

from __future__ import annotations

from pathlib import Path

from quant_risk.prod.bundle_manifest import BundleManifest


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_bundle_dir(
    bundles_dir: str | Path,
    asset_id: str,
    pointer: str = "current",
) -> Path:
    """Resolve a named pointer (``current`` or ``previous``) for *asset_id*.

    Supports two pointer conventions:
    - symbolic link pointing to the bundle version directory
    - plain text file containing the bundle version name

    Raises
    ------
    FileNotFoundError
        If the asset directory or pointer is missing.
    """
    asset_dir = Path(bundles_dir) / asset_id
    if not asset_dir.exists():
        raise FileNotFoundError(
            f"No bundle directory for asset '{asset_id}': '{asset_dir}' not found."
        )

    ptr = asset_dir / pointer
    if not ptr.exists():
        raise FileNotFoundError(
            f"Bundle pointer '{pointer}' for asset '{asset_id}' not found "
            f"(expected at '{ptr}')."
        )

    if ptr.is_symlink():
        resolved = (asset_dir / ptr.readlink()).resolve()
    elif ptr.is_dir():
        resolved = ptr.resolve()
    else:
        version = ptr.read_text(encoding="utf-8").strip()
        if not version:
            raise ValueError(
                f"Bundle pointer '{pointer}' for asset '{asset_id}' is empty."
            )
        resolved = asset_dir / version

    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Bundle directory '{resolved}' (pointed to by '{pointer}') "
            f"for asset '{asset_id}' not found."
        )
    return resolved


def get_active_bundle_dir(bundles_dir: str | Path, asset_id: str) -> Path:
    """Return the active (``current``) bundle directory for *asset_id*."""
    return get_bundle_dir(bundles_dir, asset_id, pointer="current")


def get_previous_bundle_dir(bundles_dir: str | Path, asset_id: str) -> Path | None:
    """Return the previous bundle directory for *asset_id*, or None if absent."""
    try:
        return get_bundle_dir(bundles_dir, asset_id, pointer="previous")
    except FileNotFoundError:
        return None


def has_active_bundle(bundles_dir: str | Path, asset_id: str) -> bool:
    """Return True if *asset_id* has an active (``current``) bundle."""
    try:
        get_active_bundle_dir(bundles_dir, asset_id)
        return True
    except FileNotFoundError:
        return False


def list_bundle_versions(bundles_dir: str | Path, asset_id: str) -> list[str]:
    """Return all versioned bundle directory names for *asset_id*.

    Pointer entries (``current``, ``previous``) and any entries that are not
    directories are excluded.  Results are sorted alphabetically (which is
    chronological for the ``YYYY-MM-DD_prod_<ref>`` naming convention).

    Returns an empty list if the asset directory does not exist.
    """
    asset_dir = Path(bundles_dir) / asset_id
    if not asset_dir.exists():
        return []

    _pointers = {"current", "previous"}
    versions = []
    for entry in asset_dir.iterdir():
        if entry.name in _pointers:
            continue
        # Resolve symlinks before checking is_dir()
        target = entry.resolve() if entry.is_symlink() else entry
        if target.is_dir():
            versions.append(entry.name)
    return sorted(versions)


def load_active_manifest(bundles_dir: str | Path, asset_id: str) -> BundleManifest:
    """Load and return the manifest for the active bundle of *asset_id*."""
    bundle_dir = get_active_bundle_dir(bundles_dir, asset_id)
    return BundleManifest.load(bundle_dir)
