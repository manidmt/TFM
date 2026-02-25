'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-01-10

@description: Script to build features + labels dataset.
'''

from __future__ import annotations

import argparse
import yaml

from quant_risk.features.build import BuildFeaturesConfig, build_features
from quant_risk.features.labels import BuildLabelsConfig, build_labels


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build features + labels dataset.")
    parser.add_argument("--config_features", default="config/features.yaml")
    parser.add_argument("--config_sources", default="config/datasources.yaml")
    args = parser.parse_args()

    feat_cfg = load_yaml(args.config_features)
    src_cfg = load_yaml(args.config_sources)

    db_path = src_cfg["db"]["path"]
    tickers = src_cfg["prices"]["tickers"]

    bcfg = BuildFeaturesConfig(
        db_path=db_path,
        calendar_freq=feat_cfg["calendar"]["freq"],
        rv_windows=tuple(feat_cfg["features"]["realized_vol"]["window_sizes"]),
        return_lags=tuple(feat_cfg["features"]["returns"]["lags"]),
        macro_lags=tuple(feat_cfg["features"]["macro_transforms"]["lags"]),
        macro_transform=feat_cfg["features"]["macro_transforms"]["method"],
    )
    print(build_features(bcfg, tickers=tickers))

    lcfg = BuildLabelsConfig(
        db_path=db_path,
        horizons=tuple(feat_cfg["targets"]["horizons"]),
        regime_bins=int(feat_cfg["targets"]["regime_bins"]),
        ret_col="logret",
    )
    print(build_labels(lcfg))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())