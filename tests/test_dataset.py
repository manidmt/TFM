'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-24

@description: Tests for dataset creation and integrity.
'''


import pandas as pd

from quant_risk.datasets.make_dataset import DatasetConfig, make_dataset


DB = "data/db/financial_data.duckdb"


def test_make_dataset_not_empty():
    cfg = DatasetConfig(db_path=DB, tickers=("^GSPC",), horizon=20, pooled=False)
    pack = make_dataset(cfg)

    assert pack["n_rows"] > 100, "Dataset too small; ingestion/build likely failed."
    assert pack["n_features"] > 5, "Too few features selected; feature selection may be wrong."


def test_no_leakage_features_excluded():
    cfg = DatasetConfig(db_path=DB, tickers=("^GSPC",), horizon=20, pooled=False)
    pack = make_dataset(cfg)
    cols = pack["feature_cols"]

    assert "logret" not in cols, "Leakage risk: logret should be excluded from features."
    assert not any(c.startswith("rv_") for c in cols), "Leakage risk: rv_* should be excluded from features."


def test_regime_values_in_range():
    cfg = DatasetConfig(db_path=DB, tickers=("^GSPC",), horizon=20, pooled=False)
    pack = make_dataset(cfg)
    df = pack["df"]

    assert "regime" in df.columns
    assert set(df["regime"].unique()).issubset({0, 1, 2}), "Unexpected regime labels found."


def test_time_splits_are_ordered():
    cfg = DatasetConfig(db_path=DB, tickers=("^GSPC",), horizon=20, pooled=False)
    pack = make_dataset(cfg)

    train, valid, test = pack["train"], pack["valid"], pack["test"]

    # It is possible one split is empty in edge cases; guard accordingly.
    if len(train) and len(valid):
        assert train["date"].max() < valid["date"].min()
    if len(valid) and len(test):
        assert valid["date"].max() < test["date"].min()
