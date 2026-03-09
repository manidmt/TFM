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
    gdelt_queries_cfg = src_cfg.get("gdelt", {}).get("queries", {})
    news_query_ids = (
        tuple(str(k).strip().lower() for k in gdelt_queries_cfg.keys() if str(k).strip())
        if isinstance(gdelt_queries_cfg, dict)
        else ()
    )

    bcfg = BuildFeaturesConfig(
        db_path=db_path,
        calendar_freq=feat_cfg["calendar"]["freq"],
        rv_windows=tuple(feat_cfg["features"]["realized_vol"]["window_sizes"]),
        return_lags=tuple(feat_cfg["features"]["returns"]["lags"]),
        macro_lags=tuple(feat_cfg["features"]["macro_transforms"]["lags"]),
        macro_transform=feat_cfg["features"]["macro_transforms"]["method"],
        macro_publication_lags=feat_cfg.get("features", {}).get("macro_publication_lags"),
        rv_long_window=int(feat_cfg.get("features", {}).get("shock_features", {}).get("rv_long_window", 60)),
        rv_ema_spans=tuple(feat_cfg.get("features", {}).get("shock_features", {}).get("rv_ema_spans", [10, 30])),
        vol_of_vol_window=int(feat_cfg.get("features", {}).get("shock_features", {}).get("vol_of_vol_window", 20)),
        return_shock_window=int(feat_cfg.get("features", {}).get("shock_features", {}).get("return_shock_window", 60)),
        return_shock_quantiles=tuple(feat_cfg.get("features", {}).get("shock_features", {}).get("return_shock_quantiles", [0.8, 0.9])),
        volume_z_window=int(feat_cfg.get("features", {}).get("shock_features", {}).get("volume_z_window", 20)),
        cross_corr_window=int(feat_cfg.get("features", {}).get("shock_features", {}).get("cross_corr_window", 20)),
        gdelt_table=str(src_cfg.get("gdelt", {}).get("table", "gdelt_gkg_daily")),
        news_enabled=bool(feat_cfg.get("news_features", {}).get("enabled", False)),
        news_publication_lag_bdays=int(src_cfg.get("gdelt", {}).get("publication_lag_bdays", 1)),
        news_windows=tuple(feat_cfg.get("news_features", {}).get("windows", [3, 10, 20])),
        news_include_roll_sum=bool(feat_cfg.get("news_features", {}).get("include_roll_sum", True)),
        news_include_roll_mean=bool(feat_cfg.get("news_features", {}).get("include_roll_mean", True)),
        news_include_roll_std=bool(feat_cfg.get("news_features", {}).get("include_roll_std", True)),
        news_include_tone_std=bool(feat_cfg.get("news_features", {}).get("include_tone_std", False)),
        news_include_tone_neg_share=bool(feat_cfg.get("news_features", {}).get("include_tone_neg_share", False)),
        news_query_ids=news_query_ids,
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
