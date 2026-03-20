'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-03-20

@description: Tests for price fetching helpers.
'''

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_risk.data.fetcher import PriceIngestConfig, refresh_prices


def test_refresh_prices_marks_empty_download_as_error(monkeypatch, tmp_path: Path):
    def _fake_download(_tickers, start, end, auto_adjust, timeout_seconds=30):
        return pd.DataFrame(
            columns=["ticker", "date", "open", "high", "low", "close", "volume"]
        )

    monkeypatch.setattr("quant_risk.data.fetcher.download_ohlcv", _fake_download)

    cfg = PriceIngestConfig(db_path=str(tmp_path / "prices.duckdb"))
    res = refresh_prices(cfg, tickers=["^STOXX50E"])

    assert res["inserted_rows"] == 0
    assert "^STOXX50E" in res["errors"]
    assert "Empty download result" in res["errors"]["^STOXX50E"]
