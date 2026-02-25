'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-24

@description: Quick smoke tests for baseline models training and prediction.
'''

from quant_risk.datasets.make_dataset import DatasetConfig, make_dataset, build_xy
from quant_risk.models.baseline import make_models


DB = "data/db/financial_data.duckdb"


def test_baseline_models_train_and_predict():
    cfg = DatasetConfig(db_path=DB, tickers=("^GSPC",), horizon=20, pooled=False)
    pack = make_dataset(cfg)

    feature_cols = pack["feature_cols"]
    train = pack["train"]
    valid = pack["valid"]

    # Use small subsets to keep tests fast
    train = train.tail(300)
    valid = valid.head(100)

    Xtr, ytr = build_xy(train, feature_cols)
    Xva, yva = build_xy(valid, feature_cols)

    models = make_models(random_state=7)

    models.logit.fit(Xtr, ytr)
    p1 = models.logit.predict(Xva)

    models.rf.fit(Xtr, ytr)
    p2 = models.rf.predict(Xva)

    assert len(p1) == len(yva)
    assert len(p2) == len(yva)

    assert set(p1).issubset({0, 1, 2})
    assert set(p2).issubset({0, 1, 2})
