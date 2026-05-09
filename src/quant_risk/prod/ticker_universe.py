'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-04-11

@description: Ticker universe catalog loader for portfolio autocomplete.

Reads config/prod/ticker_universe.yaml and exposes typed TickerInfo objects.
Each entry maps a tradeable ticker to one of the six production proxy assets,
enabling automatic proxy assignment in the portfolio editor.
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from quant_risk.config import load_yaml
from quant_risk.prod.assets import PRODUCTION_ASSET_IDS

_DEFAULT_CATALOG_PATH = Path(__file__).parents[3] / "config" / "prod" / "ticker_universe.yaml"

VALID_ASSET_CLASSES = frozenset({
    "us_equity",
    "euro_equity",
    "crypto",
    "bond_long",
    "bond_short",
    "commodity",
})


@dataclass(frozen=True)
class TickerInfo:
    """Typed representation of one entry in config/prod/ticker_universe.yaml."""

    ticker: str
    display_name: str
    asset_class: str
    proxy_asset_id: str

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("ticker must be a non-empty string.")
        if not self.display_name:
            raise ValueError(f"display_name for '{self.ticker}' must be non-empty.")
        if self.asset_class not in VALID_ASSET_CLASSES:
            raise ValueError(
                f"asset_class '{self.asset_class}' for '{self.ticker}' is not valid. "
                f"Allowed: {sorted(VALID_ASSET_CLASSES)}"
            )
        if self.proxy_asset_id not in PRODUCTION_ASSET_IDS:
            raise ValueError(
                f"proxy_asset_id '{self.proxy_asset_id}' for '{self.ticker}' is not valid. "
                f"Allowed: {sorted(PRODUCTION_ASSET_IDS)}"
            )


def load_ticker_universe(
    path: str | Path | None = None,
) -> list[TickerInfo]:
    """Load and validate the ticker universe catalog.

    Parameters
    ----------
    path:
        Path to the YAML catalog file. Defaults to config/prod/ticker_universe.yaml
        relative to the repo root.

    Returns
    -------
    list[TickerInfo]
        List of all tickers in the universe.

    Raises
    ------
    ValueError
        If the catalog is missing required fields, contains duplicate tickers,
        or contains invalid asset_class / proxy_asset_id values.
    """
    resolved = Path(path) if path is not None else _DEFAULT_CATALOG_PATH
    raw = load_yaml(str(resolved))

    if not isinstance(raw, dict) or "tickers" not in raw:
        raise ValueError(f"Catalog at '{resolved}' must contain a top-level 'tickers' list.")

    entries = raw["tickers"]
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError(f"Catalog at '{resolved}': 'tickers' must be a non-empty list.")

    required_fields = {"ticker", "display_name", "asset_class", "proxy_asset_id"}
    seen: set[str] = set()
    catalog: list[TickerInfo] = []

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Catalog entry #{i} is not a mapping.")
        missing = required_fields - entry.keys()
        if missing:
            raise ValueError(
                f"Catalog entry #{i} is missing required fields: {sorted(missing)}"
            )
        ticker = str(entry["ticker"])
        if ticker in seen:
            raise ValueError(f"Duplicate ticker '{ticker}' in catalog.")
        seen.add(ticker)
        catalog.append(
            TickerInfo(
                ticker=ticker,
                display_name=str(entry["display_name"]),
                asset_class=str(entry["asset_class"]),
                proxy_asset_id=str(entry["proxy_asset_id"]),
            )
        )

    return catalog


def get_ticker_info(ticker: str, catalog: Sequence[TickerInfo]) -> TickerInfo:
    """Return the TickerInfo for *ticker*, raising KeyError if not found."""
    for t in catalog:
        if t.ticker == ticker:
            return t
    raise KeyError(
        f"ticker '{ticker}' not found in universe. "
        f"Known tickers: {[t.ticker for t in catalog[:10]]}..."
    )
