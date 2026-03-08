'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-27

@description: Quick smoke tests for tabular models.
'''

from __future__ import annotations

import numpy as np
import pytest

from quant_risk.datasets.make_dataset import DatasetConfig, build_xy, make_dataset
from quant_risk.models.tabular.xgb import XGBConfig, fit as fit_xgb, make_model as make_xgb_model, predict as predict_xgb
from quant_risk.models.tabular.tabnet import TabNetConfig, fit as fit_tabnet, make_model as make_tabnet_model, predict as predict_tabnet
from quant_risk.models.tabular.ft_transformer import FTTransformerConfig, fit as fit_ftt, make_model as make_ftt_model, predict as predict_ftt
from quant_risk.models.tabular.tabpfn import (
    TabPFNConfig,
    _sanitize_tabpfn_array,
    fit as fit_tabpfn,
    make_model as make_tabpfn_model,
    predict as predict_tabpfn,
)


DB = "data/db/financial_data.duckdb"


def _small_split():
    cfg = DatasetConfig(db_path=DB, tickers=("^GSPC",), horizon=20, pooled=False)
    pack = make_dataset(cfg)
    feature_cols = pack["feature_cols"]
    train = pack["train"].tail(400)
    valid = pack["valid"].head(200)
    test = pack["test"].head(200)
    x_train, y_train = build_xy(train, feature_cols)
    x_valid, y_valid = build_xy(valid, feature_cols)
    x_test, y_test = build_xy(test, feature_cols)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def test_xgb_smoke():
    pytest.importorskip("xgboost")
    x_train, y_train, x_valid, y_valid, x_test, y_test = _small_split()

    cfg = XGBConfig(n_estimators=40, max_depth=3, learning_rate=0.1, early_stopping_rounds=10, seed=7)
    model = make_xgb_model(cfg)
    fit_xgb(model, x_train, y_train, X_valid=x_valid, y_valid=y_valid)
    pred = predict_xgb(model, x_test)

    assert pred.shape[0] == y_test.shape[0]
    assert set(pred).issubset({0, 1, 2})


def test_tabnet_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_tabnet")
    x_train, y_train, x_valid, y_valid, x_test, y_test = _small_split()

    cfg = TabNetConfig(
        n_d=8,
        n_a=8,
        n_steps=3,
        max_epochs=3,
        patience=2,
        batch_size=64,
        virtual_batch_size=16,
        seed=7,
    )
    model = make_tabnet_model(cfg)
    fit_tabnet(model, x_train, y_train, X_valid=x_valid, y_valid=y_valid)
    pred = predict_tabnet(model, x_test)

    assert pred.shape[0] == y_test.shape[0]
    assert set(pred).issubset({0, 1, 2})


def test_fttransformer_smoke():
    pytest.importorskip("torch")
    x_train, y_train, x_valid, y_valid, x_test, y_test = _small_split()

    cfg = FTTransformerConfig(
        d_token=16,
        n_heads=4,
        n_layers=1,
        ffn_multiplier=2,
        dropout=0.1,
        batch_size=64,
        max_epochs=3,
        patience=2,
        seed=7,
    )
    model = make_ftt_model(cfg)
    fit_ftt(model, x_train, y_train, X_valid=x_valid, y_valid=y_valid)
    pred = predict_ftt(model, x_test)

    assert pred.shape[0] == y_test.shape[0]
    assert set(pred).issubset({0, 1, 2})


def test_tabpfn_smoke(monkeypatch):
    monkeypatch.setenv("TABPFN_MODEL_CACHE_DIR", "/tmp/tabpfn_cache")
    pytest.importorskip("tabpfn")
    x_train, y_train, x_valid, y_valid, x_test, y_test = _small_split()

    cfg = TabPFNConfig(
        n_estimators=1,
        n_preprocessing_jobs=1,
        random_state=7,
        seed=7,
    )
    model = make_tabpfn_model(cfg)
    try:
        fit_tabpfn(model, x_train, y_train, X_valid=x_valid, y_valid=y_valid)
    except (RuntimeError, PermissionError) as e:
        err = str(e)
        if (
            "HuggingFace download failed" in err
            or "Temporary failure in name resolution" in err
            or "Cannot send a request" in err
            or "Permission denied" in err
        ):
            pytest.skip(f"TabPFN no disponible en este entorno de ejecución: {err}")
        raise
    pred = predict_tabpfn(model, x_test)

    assert pred.shape[0] == y_test.shape[0]
    assert set(pred).issubset({0, 1, 2})


def test_tabpfn_sanitize_array_handles_non_finite_and_extreme_values():
    x_train = np.array(
        [
            [0.1, np.inf, -np.inf, np.nan],
            [1.0, 2.0, 3.0, 4.0],
            [1e300, -1e300, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0],
        ],
        dtype=np.float64,
    )
    x_clean, stats = _sanitize_tabpfn_array(x_train)

    assert x_clean.dtype == np.float32
    assert np.isfinite(x_clean).all()
    assert np.max(np.abs(x_clean)) <= 1e6

    x_test = np.array([[np.inf, np.nan, -np.inf, 1e400]], dtype=np.float64)
    x_test_clean, _ = _sanitize_tabpfn_array(x_test, stats=stats)
    assert x_test_clean.dtype == np.float32
    assert np.isfinite(x_test_clean).all()
    assert np.max(np.abs(x_test_clean)) <= 1e6
