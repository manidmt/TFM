'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-27

@description: Integration test for tabular dataset+training+artifact saving.
'''

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_risk.datasets.make_dataset import DatasetConfig, build_xy, make_dataset
from quant_risk.models.metrics import compute_metrics
from quant_risk.models.tabular.common import build_variant_name
from quant_risk.models.tabular.xgb import XGBConfig, fit, make_model, predict


DB = "data/db/financial_data.duckdb"


def test_tabular_pipeline_and_artifacts(tmp_path: Path):
    pytest.importorskip("xgboost")

    cfg = DatasetConfig(db_path=DB, tickers=("^GSPC",), horizon=20, pooled=False)
    pack = make_dataset(cfg)

    feature_cols = pack["feature_cols"]
    train = pack["train"].tail(350)
    valid = pack["valid"].head(150)
    test = pack["test"].head(150)

    x_train, y_train = build_xy(train, feature_cols)
    x_valid, y_valid = build_xy(valid, feature_cols)
    x_test, y_test = build_xy(test, feature_cols)

    model = make_model(
        XGBConfig(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.1,
            random_state=7,
            seed=7,
        )
    )
    fit(model, x_train, y_train, X_valid=x_valid, y_valid=y_valid)

    pred_valid = predict(model, x_valid)
    pred_test = predict(model, x_test)
    m_valid = compute_metrics(y_valid, pred_valid)
    m_test = compute_metrics(y_test, pred_test)

    variant = build_variant_name(pooled=False, use_sarimax=False, use_garch=False)
    outdir = tmp_path / "runs" / "tabular" / "xgb" / "h20"
    variant_dir = outdir / f"variant_{variant}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "variant": variant,
        "dataset": {
            "tickers": ["^GSPC"],
            "horizon": 20,
            "pooled": False,
            "feature_cols": feature_cols,
            "bins": pack["bins"].to_dict(orient="index"),
        },
        "model": {"name": "xgb"},
    }
    metrics = {
        "valid": {"accuracy": m_valid.accuracy, "macro_f1": m_valid.macro_f1},
        "test": {"accuracy": m_test.accuracy, "macro_f1": m_test.macro_f1},
    }
    with open(variant_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(variant_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pd.DataFrame(
        [
            {
                "variant": variant,
                "model": "xgb",
                "valid_acc": m_valid.accuracy,
                "valid_macro_f1": m_valid.macro_f1,
                "test_acc": m_test.accuracy,
                "test_macro_f1": m_test.macro_f1,
                "n_features": len(feature_cols),
            }
        ]
    ).to_csv(outdir / "summary.csv", index=False)

    assert (variant_dir / "config.json").exists()
    assert (variant_dir / "metrics.json").exists()
    assert (outdir / "summary.csv").exists()
