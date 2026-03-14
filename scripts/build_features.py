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
    gkg_cfg = src_cfg.get("gkg", {})
    gdelt_cfg = src_cfg.get("gdelt", {})
    gkg_topics_cfg = gkg_cfg.get("topics", {})
    gdelt_queries_cfg = gdelt_cfg.get("queries", {})
    gkg_enabled = bool(gkg_cfg.get("enabled", False))
    gdelt_enabled = bool(gdelt_cfg.get("enabled", False))
    if gkg_enabled and isinstance(gkg_topics_cfg, dict) and gkg_topics_cfg:
        query_map = gkg_topics_cfg
    elif gdelt_enabled and isinstance(gdelt_queries_cfg, dict) and gdelt_queries_cfg:
        query_map = gdelt_queries_cfg
    else:
        query_map = {}
    news_query_ids = (
        tuple(str(k).strip().lower() for k in query_map.keys() if str(k).strip())
        if isinstance(query_map, dict)
        else ()
    )

    news_cfg = feat_cfg.get("news_features", {})
    topics_enabled_cfg = news_cfg.get("topics_enabled", [])
    if isinstance(topics_enabled_cfg, (list, tuple)):
        topics_enabled = tuple(str(x).strip().lower() for x in topics_enabled_cfg if str(x).strip())
        if topics_enabled:
            news_query_ids = topics_enabled

    source_table_override = str(news_cfg.get("source_table", "")).strip()
    preferred_news_table = (
        source_table_override
        or str(gkg_cfg.get("daily_table", "")).strip()
        or "news_features_daily"
    )
    legacy_news_table = str(gdelt_cfg.get("table", "gdelt_gkg_daily"))
    news_publication_lag_bdays = int(
        gkg_cfg.get(
            "publication_lag_bdays",
            gdelt_cfg.get("publication_lag_bdays", 1),
        )
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
        news_source_table=preferred_news_table,
        gdelt_table=legacy_news_table,
        news_enabled=bool(news_cfg.get("enabled", False)),
        news_publication_lag_bdays=news_publication_lag_bdays,
        news_windows=tuple(news_cfg.get("windows", [3, 10, 20])),
        news_include_roll_sum=bool(news_cfg.get("include_roll_sum", True)),
        news_include_roll_mean=bool(news_cfg.get("include_roll_mean", True)),
        news_include_roll_std=bool(news_cfg.get("include_roll_std", True)),
        news_include_tone_std=bool(news_cfg.get("include_tone_std", False)),
        news_include_tone_neg_share=bool(news_cfg.get("include_tone_neg_share", False)),
        news_attn_z_window=int(news_cfg.get("attn_z_window", 252)),
        news_attn_shock_threshold=float(news_cfg.get("attn_shock_threshold", 2.0)),
        news_shock_windows=tuple(news_cfg.get("shock_windows", [3, 10, 20])),
        news_compact_mode=bool(news_cfg.get("compact_mode", False)),
        news_topic_share_enabled=bool(
            news_cfg.get("topic_share_enabled", True)
        ),
        news_health_min_nonzero_rate=float(
            news_cfg.get("health", {}).get("min_nonzero_rate", 0.01)
        ),
        news_health_min_unique_values=int(
            news_cfg.get("health", {}).get("min_unique_values", 3)
        ),
        news_health_drop_dead_queries=bool(
            news_cfg.get("health", {}).get("drop_dead_queries", True)
        ),
        news_health_fail_if_all_dead=bool(
            news_cfg.get("health", {}).get("fail_if_all_dead", True)
        ),
        news_include_interactions=bool(news_cfg.get("include_interactions", True)),
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
