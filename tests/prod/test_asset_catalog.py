'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-28

@description: Tests for the production asset catalog loader (src/quant_risk/prod/assets.py).

Covers:
- Canonical catalog loads all 6 production assets with all required fields.
- No duplicate asset_ids in the catalog.
- All source_tickers match the expected production universe.
- AssetInfo is immutable (frozen dataclass).
- get_asset_by_id and get_ticker_for_asset happy path and error path.
- ticker_to_asset_id happy path and error path.
- load_asset_catalog rejects malformed YAML structures.
- load_asset_catalog rejects invalid category values.
- load_asset_catalog rejects missing required fields.
- load_asset_catalog rejects duplicate asset_ids.
'''

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from quant_risk.prod.assets import (
    PRODUCTION_ASSET_IDS,
    AssetInfo,
    get_asset_by_id,
    get_ticker_for_asset,
    load_asset_catalog,
    ticker_to_asset_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_catalog(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "assets.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Canonical catalog
# ---------------------------------------------------------------------------

def test_canonical_catalog_loads_six_assets():
    catalog = load_asset_catalog()
    assert len(catalog) == 6


def test_canonical_catalog_contains_all_production_asset_ids():
    catalog = load_asset_catalog()
    loaded_ids = {a.asset_id for a in catalog}
    assert loaded_ids == PRODUCTION_ASSET_IDS


def test_canonical_catalog_all_fields_non_empty():
    catalog = load_asset_catalog()
    for asset in catalog:
        assert asset.asset_id
        assert asset.display_name
        assert asset.source_ticker
        assert asset.category
        assert asset.short_description


def test_canonical_catalog_expected_tickers():
    expected = {"^GSPC", "^STOXX50E", "BTC-USD", "TLT", "SHY", "GLD"}
    catalog = load_asset_catalog()
    assert {a.source_ticker for a in catalog} == expected


def test_canonical_catalog_no_duplicate_asset_ids():
    catalog = load_asset_catalog()
    ids = [a.asset_id for a in catalog]
    assert len(ids) == len(set(ids))


def test_canonical_catalog_valid_categories():
    from quant_risk.prod.assets import VALID_CATEGORIES
    catalog = load_asset_catalog()
    for asset in catalog:
        assert asset.category in VALID_CATEGORIES


def test_asset_info_is_immutable():
    catalog = load_asset_catalog()
    asset = catalog[0]
    with pytest.raises((AttributeError, TypeError)):
        asset.asset_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Specific asset_id ↔ ticker mappings (guard against accidental edits)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("asset_id,expected_ticker", [
    ("us_equities",        "^GSPC"),
    ("euro_equities",      "^STOXX50E"),
    ("bitcoin",            "BTC-USD"),
    ("long_us_treasuries", "TLT"),
    ("short_us_treasuries","SHY"),
    ("gold",               "GLD"),
])
def test_ticker_mapping_per_asset(asset_id, expected_ticker):
    catalog = load_asset_catalog()
    assert get_ticker_for_asset(asset_id, catalog) == expected_ticker


# ---------------------------------------------------------------------------
# get_asset_by_id
# ---------------------------------------------------------------------------

def test_get_asset_by_id_returns_correct_asset():
    catalog = load_asset_catalog()
    asset = get_asset_by_id("bitcoin", catalog)
    assert asset.asset_id == "bitcoin"
    assert asset.source_ticker == "BTC-USD"


def test_get_asset_by_id_raises_key_error_for_unknown():
    catalog = load_asset_catalog()
    with pytest.raises(KeyError, match="unknown_asset"):
        get_asset_by_id("unknown_asset", catalog)


# ---------------------------------------------------------------------------
# ticker_to_asset_id
# ---------------------------------------------------------------------------

def test_ticker_to_asset_id_happy_path():
    catalog = load_asset_catalog()
    assert ticker_to_asset_id("^GSPC", catalog) == "us_equities"
    assert ticker_to_asset_id("GLD", catalog) == "gold"


def test_ticker_to_asset_id_raises_for_unknown_ticker():
    catalog = load_asset_catalog()
    with pytest.raises(KeyError, match="AAPL"):
        ticker_to_asset_id("AAPL", catalog)


# ---------------------------------------------------------------------------
# load_asset_catalog with custom paths / malformed content
# ---------------------------------------------------------------------------

def test_load_catalog_from_custom_path(tmp_path):
    content = """
    version: "1.0.0"
    assets:
      - asset_id: test_equity
        display_name: "Test Equity"
        source_ticker: "TEST"
        category: Equity
        short_description: "A test asset."
    """
    p = _write_catalog(tmp_path, content)
    catalog = load_asset_catalog(p)
    assert len(catalog) == 1
    assert catalog[0].asset_id == "test_equity"
    assert catalog[0].source_ticker == "TEST"


def test_load_catalog_rejects_missing_assets_key(tmp_path):
    content = "version: '1.0.0'\n"
    p = _write_catalog(tmp_path, content)
    with pytest.raises(ValueError, match="'assets'"):
        load_asset_catalog(p)


def test_load_catalog_rejects_empty_assets_list(tmp_path):
    content = "assets: []\n"
    p = _write_catalog(tmp_path, content)
    with pytest.raises(ValueError, match="non-empty"):
        load_asset_catalog(p)


def test_load_catalog_rejects_missing_required_field(tmp_path):
    content = """
    assets:
      - asset_id: test_equity
        display_name: "Test Equity"
        source_ticker: "TEST"
        category: Equity
        # short_description intentionally omitted
    """
    p = _write_catalog(tmp_path, content)
    with pytest.raises(ValueError, match="short_description"):
        load_asset_catalog(p)


def test_load_catalog_rejects_duplicate_asset_id(tmp_path):
    content = """
    assets:
      - asset_id: dupe
        display_name: "Dupe A"
        source_ticker: "DUPEA"
        category: Equity
        short_description: "First."
      - asset_id: dupe
        display_name: "Dupe B"
        source_ticker: "DUPEB"
        category: Equity
        short_description: "Second."
    """
    p = _write_catalog(tmp_path, content)
    with pytest.raises(ValueError, match="Duplicate asset_id"):
        load_asset_catalog(p)


def test_load_catalog_rejects_invalid_category(tmp_path):
    content = """
    assets:
      - asset_id: bad_cat
        display_name: "Bad Category"
        source_ticker: "BAD"
        category: Unknown
        short_description: "Invalid category."
    """
    p = _write_catalog(tmp_path, content)
    with pytest.raises(ValueError, match="Unknown"):
        load_asset_catalog(p)
