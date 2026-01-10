'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-10

@description: Module to build regime volatility labels (eg. high/medium/low [2, 1, 0]) from features.
'''

from __future__ import annotations
from dataclasses import dataclass

import duckdb
import pandas as pd

@dataclass(frozen=True)
class BuildLabelsConfig:
    db_path: str
    features_table: str = "features_daily"
    out_table: str = "labels_regime"
    horizons: tuple[int, ...] = (5, 20)
    regime_bins: int = 3
    vol_col: str = "rv_20"

def build_labels(cfg: BuildLabelsConfig) -> dict:
    con = duckdb.connect(cfg.db_path)
    try:
        df = con.execute(
            f"SELECT ticker, date, {cfg.vol_col} AS vol FROM {cfg.features_table} ORDER BY ticker, date"
        ).df()
        if df.empty:
            raise RuntimeError("features_daily is empty; run build_features first.")

        df["date"] = pd.to_datetime(df["date"]).dt.date

        out_all = []
        for h in cfg.horizons:
            tmp = df.copy()
            # forward-looking label: vol at t+h
            tmp["vol_fwd"] = tmp.groupby("ticker")["vol"].shift(-h)

            # quantile bins per ticker (avoids scale issues)
            def _bin(s: pd.Series) -> pd.Series:
                return pd.qcut(s, q=cfg.regime_bins, labels=False, duplicates="drop")

            tmp["regime"] = tmp.groupby("ticker")["vol_fwd"].transform(_bin)

            tmp["horizon"] = h
            tmp = tmp.dropna(subset=["vol_fwd", "regime"])
            tmp["regime"] = tmp["regime"].astype(int)

            out_all.append(tmp[["ticker", "date", "horizon", "regime", "vol_fwd"]])

        out = pd.concat(out_all, ignore_index=True)

        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {cfg.out_table} (
              ticker TEXT,
              date DATE,
              horizon INTEGER,
              regime INTEGER,
              vol_fwd DOUBLE
            )
        """)
        con.register("tmp_labels", out)
        con.execute(f"DELETE FROM {cfg.out_table}")
        con.execute(f"INSERT INTO {cfg.out_table} SELECT * FROM tmp_labels")

        return {"rows": len(out), "out_table": cfg.out_table, "horizons": list(cfg.horizons)}
    finally:
        con.close()
