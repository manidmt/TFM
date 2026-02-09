'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-10

@description: Train hybrid volatility-regime classifiers with SARIMAX + tabular features.
'''

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

from quant_risk.datasets.make_dataset import DatasetConfig, build_xy, make_dataset


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, model_name: str, model):
    """Train model and print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    print("\nTrain Metrics:")
    print(f"  Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  F1 (weighted): {f1_score(y_train, y_train_pred, average='weighted'):.4f}")
    
    print("\nValidation Metrics:")
    print(f"  Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"  F1 (weighted): {f1_score(y_val, y_val_pred, average='weighted'):.4f}")
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred))
    
    print("\nTest Metrics:")
    print(f"  Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"  F1 (weighted): {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train hybrid SARIMAX + tabular models for regime classification."
    )
    parser.add_argument("--config_dataset", default="config/dataset.yaml")
    parser.add_argument("--config_sources", default="config/datasources.yaml")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--sarimax_exog", nargs="*", default=["rv_5", "rv_20"])
    args = parser.parse_args()
    
    ds_cfg = load_yaml(args.config_dataset)
    src_cfg = load_yaml(args.config_sources)
    
    db_path = src_cfg["db"]["path"]
    
    # Enable SARIMAX features
    dataset_cfg = DatasetConfig(
        db_path=db_path,
        horizon=args.horizon,
        train_end=ds_cfg["split"]["train_end"],
        val_end=ds_cfg["split"]["val_end"],
        regime_bins=ds_cfg["targets"]["regime_bins"],
        use_sarimax=True,
        sarimax_order=(1, 0, 1),
        sarimax_seasonal_order=(0, 0, 0, 0),
        sarimax_trend="c",
        sarimax_log_transform=True,
        sarimax_exog_cols=tuple(args.sarimax_exog),
    )
    
    print("Building datasets with SARIMAX features...")
    pack = make_dataset(dataset_cfg)
    train, valid, test = pack["train"], pack["valid"], pack["test"]
    feature_cols = pack["feature_cols"]

    X_train, y_train = build_xy(train, feature_cols)
    X_val, y_val = build_xy(valid, feature_cols)
    X_test, y_test = build_xy(test, feature_cols)
        
    print(f"\nDataset shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val: X={X_val.shape}, y={y_val.shape}")
    print(f"  Test: X={X_test.shape}, y={y_test.shape}")
    print(f"\nFeature columns (first 10): {feature_cols[:10]}")
    print(f"Total features: {len(feature_cols)}")
    
    # Check for SARIMAX features
    sarimax_feats = [c for c in feature_cols if c.startswith('sarimax_')]
    max_preview = 20
    print(f"\nSARIMAX features included: {len(sarimax_feats)} (showing up to {max_preview}):")
    for feat in sarimax_feats[:max_preview]:
        print(f"  - {feat}")
    if len(sarimax_feats) > max_preview:
        print("  ...")
    
    # Train Logistic Regression
    lr_model = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        "Logistic Regression (Hybrid)",
        LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    )
    
    # Train Random Forest
    rf_model = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        "Random Forest (Hybrid)",
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    )
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
