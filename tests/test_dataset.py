'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-24

@description: Tests for dataset creation and integrity.
'''


import pandas as pd
import numpy as np

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


def test_make_dataset_pooled_keeps_split_logic():
    cfg = DatasetConfig(db_path=DB, tickers=("^GSPC",), horizon=20, pooled=True)
    pack = make_dataset(cfg)
    assert len(pack["train"]) > 0
    assert len(pack["valid"]) > 0
    assert len(pack["test"]) > 0


def test_feature_selection_uses_train_only(monkeypatch):
    dates = pd.bdate_range("2020-01-01", periods=120)
    horizon = 5
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "ticker": "AAA",
                "date": d,
                "logret": 0.01 * np.sin(i / 5.0),
                "logret_lag1": 0.01 * np.sin((i - 1) / 5.0) if i > 0 else 0.0,
                "candidate_train_only": float(i),
                "vol_fwd": 0.05 + 0.001 * i,
            }
        )
    df = pd.DataFrame(rows)
    train_end = pd.Timestamp(dates[79])
    valid_end = pd.Timestamp(dates[104])
    df.loc[df["date"] > train_end, "candidate_train_only"] = np.nan

    def _fake_load_joined(_cfg):
        return df.copy()

    monkeypatch.setattr("quant_risk.datasets.make_dataset.load_joined", _fake_load_joined)

    cfg = DatasetConfig(
        db_path=DB,
        tickers=("AAA",),
        horizon=horizon,
        pooled=False,
        train_end=str(train_end.date()),
        valid_end=str(valid_end.date()),
    )
    pack = make_dataset(cfg)

    assert "candidate_train_only" in pack["feature_cols"]


def test_sarimax_exog_rejects_future_label_column():
    cfg = DatasetConfig(
        db_path=DB,
        tickers=("^GSPC",),
        horizon=20,
        sarimax_exog_cols=("vol_fwd",),
    )
    try:
        make_dataset(cfg)
        raise AssertionError("Expected ValueError when vol_fwd is used in sarimax_exog_cols")
    except ValueError as e:
        assert "sarimax_exog_cols" in str(e)
        assert "vol_fwd" in str(e)
