'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Production asset catalog loader.

Reads config/prod/assets.yaml and exposes typed AssetInfo objects.
The catalog is the single source of truth for the six production proxies —
asset_id is the stable semantic identifier used across the entire production
stack (bundles, serving.duckdb, API, frontend).

Reference: rpi5.md §4, §26.1, §26.4
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from quant_risk.config import load_yaml

_DEFAULT_CATALOG_PATH = Path(__file__).parents[3] / "config" / "prod" / "assets.yaml"

# Valid categories defined in the production universe (rpi5.md §4).
VALID_CATEGORIES = frozenset({"Equity", "Crypto", "Rates", "Commodity"})

# The canonical six production asset_ids (rpi5.md §4).
PRODUCTION_ASSET_IDS = frozenset({
    "us_equities",
    "euro_equities",
    "bitcoin",
    "long_us_treasuries",
    "short_us_treasuries",
    "gold",
})


@dataclass(frozen=True)
class AssetInfo:
    """Typed representation of one entry in config/prod/assets.yaml."""

    asset_id: str
    display_name: str
    source_ticker: str
    category: str
    short_description: str

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id must be a non-empty string.")
        if not self.source_ticker:
            raise ValueError(f"source_ticker for '{self.asset_id}' must be non-empty.")
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"category '{self.category}' for '{self.asset_id}' is not valid. "
                f"Allowed: {sorted(VALID_CATEGORIES)}"
            )


def load_asset_catalog(
    path: str | Path | None = None,
) -> list[AssetInfo]:
    """Load and validate the production asset catalog.

    Parameters
    ----------
    path:
        Path to the YAML catalog file. Defaults to config/prod/assets.yaml
        relative to the repo root.

    Returns
    -------
    list[AssetInfo]
        Ordered list of all production assets.

    Raises
    ------
    ValueError
        If the catalog is missing required fields, contains duplicate asset_ids,
        or contains an invalid category.
    """
    resolved = Path(path) if path is not None else _DEFAULT_CATALOG_PATH
    raw = load_yaml(str(resolved))

    if not isinstance(raw, dict) or "assets" not in raw:
        raise ValueError(f"Catalog at '{resolved}' must contain a top-level 'assets' list.")

    entries = raw["assets"]
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError(f"Catalog at '{resolved}': 'assets' must be a non-empty list.")

    required_fields = {"asset_id", "display_name", "source_ticker", "category", "short_description"}
    seen_ids: set[str] = set()
    catalog: list[AssetInfo] = []

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Catalog entry #{i} is not a mapping.")
        missing = required_fields - entry.keys()
        if missing:
            raise ValueError(
                f"Catalog entry #{i} is missing required fields: {sorted(missing)}"
            )
        asset_id = str(entry["asset_id"])
        if asset_id in seen_ids:
            raise ValueError(f"Duplicate asset_id '{asset_id}' in catalog.")
        seen_ids.add(asset_id)
        catalog.append(
            AssetInfo(
                asset_id=asset_id,
                display_name=str(entry["display_name"]),
                source_ticker=str(entry["source_ticker"]),
                category=str(entry["category"]),
                short_description=str(entry["short_description"]),
            )
        )

    return catalog


def get_asset_by_id(asset_id: str, catalog: Sequence[AssetInfo]) -> AssetInfo:
    """Return the AssetInfo for *asset_id*, raising KeyError if not found."""
    for asset in catalog:
        if asset.asset_id == asset_id:
            return asset
    raise KeyError(
        f"asset_id '{asset_id}' not found in catalog. "
        f"Known ids: {[a.asset_id for a in catalog]}"
    )


def get_ticker_for_asset(asset_id: str, catalog: Sequence[AssetInfo]) -> str:
    """Return the source_ticker for *asset_id*."""
    return get_asset_by_id(asset_id, catalog).source_ticker


def ticker_to_asset_id(ticker: str, catalog: Sequence[AssetInfo]) -> str:
    """Return the asset_id whose source_ticker matches *ticker*, raising KeyError if not found."""
    for asset in catalog:
        if asset.source_ticker == ticker:
            return asset.asset_id
    raise KeyError(
        f"ticker '{ticker}' not found in catalog. "
        f"Known tickers: {[a.source_ticker for a in catalog]}"
    )
