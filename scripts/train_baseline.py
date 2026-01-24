'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-24

@description: Script to train and evaluate baseline (Logistic Regression and Random Forest) models for regime classification.
'''


from __future__ import annotations

import argparse

from quant_risk.datasets.make_dataset import DatasetConfig, make_dataset, build_xy
from quant_risk.models.baseline import make_models
from quant_risk.models.metrics import compute_metrics, per_class_accuracy


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

    print("\n===== LOGIT (multinomial) =====")
    print(f"VALID acc={m_va.accuracy:.4f} macroF1={m_va.macro_f1:.4f}")
    print(m_va.report)
    print("Per-class acc (VALID):", per_class_accuracy(m_va.confusion))

    print(f"\nTEST  acc={m_te.accuracy:.4f} macroF1={m_te.macro_f1:.4f}")
    print(m_te.report)
    print("Per-class acc (TEST):", per_class_accuracy(m_te.confusion))

    # ---- RF ----
    models.rf.fit(Xtr, ytr)
    pred_va = models.rf.predict(Xva)
    pred_te = models.rf.predict(Xte)

    m_va = compute_metrics(yva, pred_va)
    m_te = compute_metrics(yte, pred_te)

    print("\n===== RandomForest =====")
    print(f"VALID acc={m_va.accuracy:.4f} macroF1={m_va.macro_f1:.4f}")
    print(m_va.report)
    print("Per-class acc (VALID):", per_class_accuracy(m_va.confusion))

    print(f"\nTEST  acc={m_te.accuracy:.4f} macroF1={m_te.macro_f1:.4f}")
    print(m_te.report)
    print("Per-class acc (TEST):", per_class_accuracy(m_te.confusion))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())