'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-24

@description: Script to train and evaluate baseline (Logistic Regression and Random Forest) models for regime classification.
'''


from __future__ import annotations

import argparse
from struct import pack

from quant_risk.datasets.make_dataset import DatasetConfig, make_dataset, build_xy
from quant_risk.models.baseline import make_models
from quant_risk.models.metrics import compute_metrics, per_class_accuracy


def print_model_metrics(name: str, metrics_valid, metrics_test) -> None:
    print(f"\n===== {name} =====")
    print(f"VALID acc={metrics_valid.accuracy:.4f} macroF1={metrics_valid.macro_f1:.4f}")
    print(metrics_valid.report)
    print("Per-class acc (VALID):", per_class_accuracy(metrics_valid.confusion))

    print(f"\nTEST  acc={metrics_test.accuracy:.4f} macroF1={metrics_test.macro_f1:.4f}")
    print(metrics_test.report)
    print("Per-class acc (TEST):", per_class_accuracy(metrics_test.confusion))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/financial_data.duckdb")
    ap.add_argument("--horizon", type=int, default=20, choices=[5, 20])
    ap.add_argument("--tickers", nargs="+", default=["^GSPC", "BTC-USD", "TLT"])
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--train_end", default=None)
    ap.add_argument("--valid_end", default=None)
    args = ap.parse_args()

    cfg = DatasetConfig(
        db_path=args.db,
        tickers=tuple(args.tickers),
        horizon=args.horizon,
        pooled=bool(args.pooled),
        train_end=args.train_end,
        valid_end=args.valid_end,
    )

    pack = make_dataset(cfg)
    feature_cols = pack["feature_cols"]

    train, valid, test = pack["train"], pack["valid"], pack["test"]

    print("\n--- Regime distribution ---")
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        vc = split["regime"].value_counts().sort_index()
        print(name, vc.to_dict())

    print("\n--- Train-fitted bins (q1/q2) ---")
    print(pack.get("bins"))


    Xtr, ytr = build_xy(train, feature_cols)
    Xva, yva = build_xy(valid, feature_cols)
    Xte, yte = build_xy(test, feature_cols)

    models = make_models()

    # ---- LOGIT ----
    models.logit.fit(Xtr, ytr)
    pred_va = models.logit.predict(Xva)
    pred_te = models.logit.predict(Xte)

    m_va = compute_metrics(yva, pred_va)
    m_te = compute_metrics(yte, pred_te)

    print_model_metrics("LOGIT (multinomial)", m_va, m_te)

    # ---- RF ----
    models.rf.fit(Xtr, ytr)
    pred_va = models.rf.predict(Xva)
    pred_te = models.rf.predict(Xte)

    m_va = compute_metrics(yva, pred_va)
    m_te = compute_metrics(yte, pred_te)

    print_model_metrics("RandomForest", m_va, m_te)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())