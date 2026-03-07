#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_new_run(run_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    specs = [
        ("xgb", run_root / "block1_xgb"),
        ("tabpfn", run_root / "block2_tabpfn"),
    ]
    for model, root in specs:
        for h in (5, 20):
            p = root / f"h{h}" / "final_vs_persistence.csv"
            df = _read_csv_if_exists(p)
            if df.empty:
                continue
            df = df.copy()
            df["model"] = model
            if "horizon" not in df.columns:
                df["horizon"] = int(h)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if "asset" in out.columns:
        out["asset"] = out["asset"].astype(str)
    return out


def _load_previous_xgb(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path).copy()
    df["model"] = "xgb"
    df["source_kind"] = "previous"
    return df


def _load_previous_tabpfn(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for h in (5, 20):
        p = root / f"h{h}" / "final_vs_persistence.csv"
        df = _read_csv_if_exists(p)
        if df.empty:
            continue
        df = df.copy()
        df["model"] = "tabpfn"
        if "horizon" not in df.columns:
            df["horizon"] = int(h)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["source_kind"] = "previous"
    return out


def _safe_mean(s: pd.Series) -> float:
    if s.empty:
        return float("nan")
    return float(np.nanmean(pd.to_numeric(s, errors="coerce")))


def _summarize(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = ["model", "horizon"]
    out_rows: list[dict[str, Any]] = []
    for (model, horizon), g in df.groupby(group_cols, dropna=False):
        delta_col = (
            "delta_test_macro_f1_vs_persistence"
            if "delta_test_macro_f1_vs_persistence" in g.columns
            else None
        )
        row = {
            "run_tag": tag,
            "model": str(model),
            "horizon": int(horizon),
            "n_assets": int(len(g)),
            "chain_macro_f1_mean": _safe_mean(g["chain_test_macro_f1"]) if "chain_test_macro_f1" in g.columns else float("nan"),
            "persistence_macro_f1_mean": _safe_mean(g["persistence_test_macro_f1"]) if "persistence_test_macro_f1" in g.columns else float("nan"),
            "delta_macro_f1_vs_persistence_mean": _safe_mean(g[delta_col]) if delta_col else float("nan"),
            "win_rate_vs_persistence_macro_f1": float(np.mean(pd.to_numeric(g[delta_col], errors="coerce") > 0.0)) if delta_col else float("nan"),
            "chain_acc_mean": _safe_mean(g["chain_test_acc"]) if "chain_test_acc" in g.columns else float("nan"),
            "persistence_acc_mean": _safe_mean(g["persistence_test_acc"]) if "persistence_test_acc" in g.columns else float("nan"),
        }
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(["model", "horizon"]).reset_index(drop=True)


def _join_new_vs_prev(new_s: pd.DataFrame, prev_s: pd.DataFrame) -> pd.DataFrame:
    if new_s.empty:
        return pd.DataFrame()
    if prev_s.empty:
        out = new_s.copy()
        out["prev_chain_macro_f1_mean"] = np.nan
        out["improvement_vs_prev_chain_macro_f1"] = np.nan
        out["prev_delta_macro_f1_vs_persistence_mean"] = np.nan
        out["improvement_vs_prev_delta_macro_f1"] = np.nan
        return out

    merged = new_s.merge(
        prev_s[[
            "model",
            "horizon",
            "chain_macro_f1_mean",
            "delta_macro_f1_vs_persistence_mean",
        ]].rename(
            columns={
                "chain_macro_f1_mean": "prev_chain_macro_f1_mean",
                "delta_macro_f1_vs_persistence_mean": "prev_delta_macro_f1_vs_persistence_mean",
            }
        ),
        on=["model", "horizon"],
        how="left",
    )
    merged["improvement_vs_prev_chain_macro_f1"] = (
        merged["chain_macro_f1_mean"] - merged["prev_chain_macro_f1_mean"]
    )
    merged["improvement_vs_prev_delta_macro_f1"] = (
        merged["delta_macro_f1_vs_persistence_mean"]
        - merged["prev_delta_macro_f1_vs_persistence_mean"]
    )
    return merged


def _fmt(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if np.isnan(v):
        return "nan"
    return f"{v:.4f}"


def _to_markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_Sin datos._"
    d = df[cols].copy()
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].map(_fmt)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(map(str, row)) + " |" for row in d.to_numpy())
    return "\n".join([header, sep, body])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build long-run chain report and comparisons.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument(
        "--previous_xgb_csv",
        default="runs/walk_forward_chain_xgb_night_20260301_193740/final_vs_persistence_all.csv",
    )
    parser.add_argument(
        "--previous_tabpfn_root",
        default="runs/walk_forward_chain_tabpfn_v25_best_20260305_140444/regime",
    )
    parser.add_argument("--report_path", default="")
    parser.add_argument("--report_mirror", default="")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    new_df = _load_new_run(run_root)
    if new_df.empty:
        raise SystemExit(f"No final_vs_persistence.csv found under {run_root}")
    new_df["source_kind"] = "new"

    prev_xgb = _load_previous_xgb(Path(args.previous_xgb_csv))
    prev_tab = _load_previous_tabpfn(Path(args.previous_tabpfn_root))
    prev_df = pd.concat([d for d in [prev_xgb, prev_tab] if not d.empty], ignore_index=True) if (not prev_xgb.empty or not prev_tab.empty) else pd.DataFrame()

    # Persist raw combined results
    raw_out = run_root / "final_vs_persistence_all_models.csv"
    new_df.to_csv(raw_out, index=False)

    # Summaries
    new_summary = _summarize(new_df, tag="new")
    prev_summary = _summarize(prev_df, tag="previous") if not prev_df.empty else pd.DataFrame()
    cmp_summary = _join_new_vs_prev(new_summary, prev_summary)

    by_asset_cols = [
        c
        for c in [
            "model",
            "asset",
            "horizon",
            "chain_test_macro_f1",
            "persistence_test_macro_f1",
            "delta_test_macro_f1_vs_persistence",
            "chain_test_acc",
            "persistence_test_acc",
            "delta_test_acc_vs_persistence",
        ]
        if c in new_df.columns
    ]
    by_asset = new_df[by_asset_cols].copy().sort_values(["model", "asset", "horizon"])

    new_summary.to_csv(run_root / "summary_new_mean_by_model_horizon.csv", index=False)
    if not prev_summary.empty:
        prev_summary.to_csv(run_root / "summary_previous_mean_by_model_horizon.csv", index=False)
    cmp_summary.to_csv(run_root / "summary_new_vs_previous.csv", index=False)
    by_asset.to_csv(run_root / "summary_new_by_asset.csv", index=False)

    # Build markdown report
    ts_match = re.search(r"(\d{8}_\d{6})", run_root.name)
    ts = ts_match.group(1) if ts_match else ""
    report_path = Path(args.report_path) if args.report_path else (run_root / "report.md")
    mirror_path = Path(args.report_mirror) if args.report_mirror else (
        Path("reports") / (f"informe_tanda_larga_chain_{ts}.md" if ts else "informe_tanda_larga_chain.md")
    )

    report_lines: list[str] = []
    report_lines.append("# Informe Tanda Larga Chain (XGB + TabPFN)")
    report_lines.append("")
    report_lines.append(f"- Run root: `{run_root}`")
    report_lines.append(f"- Raw resultados nuevos: `{raw_out}`")
    report_lines.append("")
    report_lines.append("## Cambios Implementados")
    report_lines.append("")
    report_lines.append("- Eliminación completa del modo `delta` y vuelta al target `regime`.")
    report_lines.append("- Variantes estructurales de chain externalizadas a YAML en `src/quant_risk/models/econometric/chain_variants.yaml`.")
    report_lines.append("- Variantes TabPFN externalizadas a YAML en `src/quant_risk/models/tabular/tabpfn_variants.yaml`.")
    report_lines.append("- `scripts/walk_forward_chain_xgb.py` actualizado para consumir ambos YAML mediante flags CLI.")
    report_lines.append("")
    report_lines.append("## Resumen Nuevo (Media por Modelo/Horizonte)")
    report_lines.append("")
    report_lines.append(
        _to_markdown_table(
            new_summary,
            [
                "model",
                "horizon",
                "n_assets",
                "chain_macro_f1_mean",
                "persistence_macro_f1_mean",
                "delta_macro_f1_vs_persistence_mean",
                "win_rate_vs_persistence_macro_f1",
                "chain_acc_mean",
                "persistence_acc_mean",
            ],
        )
    )
    report_lines.append("")
    report_lines.append("## Nuevo vs Previous")
    report_lines.append("")
    report_lines.append(
        _to_markdown_table(
            cmp_summary,
            [
                "model",
                "horizon",
                "chain_macro_f1_mean",
                "prev_chain_macro_f1_mean",
                "improvement_vs_prev_chain_macro_f1",
                "delta_macro_f1_vs_persistence_mean",
                "prev_delta_macro_f1_vs_persistence_mean",
                "improvement_vs_prev_delta_macro_f1",
            ],
        )
    )
    report_lines.append("")
    report_lines.append("## Detalle Nuevo por Activo")
    report_lines.append("")
    report_lines.append(
        _to_markdown_table(
            by_asset,
            [
                c
                for c in [
                    "model",
                    "asset",
                    "horizon",
                    "chain_test_macro_f1",
                    "persistence_test_macro_f1",
                    "delta_test_macro_f1_vs_persistence",
                    "chain_test_acc",
                    "persistence_test_acc",
                    "delta_test_acc_vs_persistence",
                ]
                if c in by_asset.columns
            ],
        )
    )
    report_lines.append("")
    report_lines.append("## Artefactos")
    report_lines.append("")
    report_lines.append(f"- `{run_root / 'summary_new_mean_by_model_horizon.csv'}`")
    if not prev_summary.empty:
        report_lines.append(f"- `{run_root / 'summary_previous_mean_by_model_horizon.csv'}`")
    report_lines.append(f"- `{run_root / 'summary_new_vs_previous.csv'}`")
    report_lines.append(f"- `{run_root / 'summary_new_by_asset.csv'}`")

    txt = "\n".join(report_lines).strip() + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(txt, encoding="utf-8")
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    mirror_path.write_text(txt, encoding="utf-8")

    print(f"[report] wrote {report_path}")
    print(f"[report] wrote {mirror_path}")
    print(f"[report] wrote {run_root / 'summary_new_vs_previous.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
