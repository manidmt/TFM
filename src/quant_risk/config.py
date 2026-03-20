'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-20

@description: Shared helpers to resolve project configuration from YAML files.
'''

from __future__ import annotations

from collections.abc import Sequence

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_price_tickers(config_path: str = "config/datasources.yaml") -> tuple[str, ...]:
    cfg = load_yaml(config_path)
    prices_cfg = cfg.get("prices", {}) if isinstance(cfg, dict) else {}
    raw_tickers = prices_cfg.get("tickers", []) if isinstance(prices_cfg, dict) else []
    tickers = tuple(str(t).strip() for t in raw_tickers if str(t).strip())
    if not tickers:
        raise ValueError(
            f"{config_path}: prices.tickers must contain at least one non-empty ticker."
        )
    return tickers


def resolve_tickers(
    explicit_tickers: Sequence[str] | None,
    config_path: str = "config/datasources.yaml",
) -> tuple[str, ...]:
    if explicit_tickers:
        tickers = tuple(str(t).strip() for t in explicit_tickers if str(t).strip())
        if tickers:
            return tickers
    return load_price_tickers(config_path)
