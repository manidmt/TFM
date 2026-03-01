#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <walkforward_root> [outdir]"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WF_ROOT="$1"
OUTDIR="${2:-runs/baseline_eval_vs_best_chain_$(date +%Y%m%d_%H%M%S)}"
WAIT_MAX_MIN="${WAIT_MAX_MIN:-300}"
WAIT_SLEEP_SEC="${WAIT_SLEEP_SEC:-60}"
XGB_N_JOBS="${XGB_N_JOBS:-1}"

ASSETS=("^GSPC" "BTC-USD" "TLT")
HORIZONS=(5 20)
EXPECTED_BEST_COUNT=$(( ${#ASSETS[@]} * ${#HORIZONS[@]} ))

echo "[INFO] Waiting for walk-forward artifacts in: ${WF_ROOT}"
echo "[INFO] Expected best.json files: ${EXPECTED_BEST_COUNT}"
echo "[INFO] Timeout (minutes): ${WAIT_MAX_MIN}"

start_ts="$(date +%s)"
while true; do
  now_ts="$(date +%s)"
  elapsed="$((now_ts - start_ts))"
  best_count="$(find "$WF_ROOT" -type f -path "*/h*/asset_*/best.json" | wc -l | tr -d ' ')"

  if [[ "$best_count" -ge "$EXPECTED_BEST_COUNT" ]] && [[ -f "$WF_ROOT/final_vs_persistence_all.csv" ]]; then
    echo "[OK] Walk-forward completed. best.json count=${best_count}"
    break
  fi

  if [[ "$elapsed" -ge "$((WAIT_MAX_MIN * 60))" ]]; then
    echo "[FAIL] Timeout waiting for walk-forward completion."
    echo "       best.json count=${best_count}, expected=${EXPECTED_BEST_COUNT}"
    exit 2
  fi

  echo "[WAIT] elapsed=${elapsed}s best.json=${best_count}/${EXPECTED_BEST_COUNT}"
  sleep "$WAIT_SLEEP_SEC"
done

mkdir -p "$OUTDIR"
echo "[INFO] Output dir: ${OUTDIR}"

for asset in "${ASSETS[@]}"; do
  safe_asset="${asset//[^a-zA-Z0-9]/_}"
  asset_out="${OUTDIR}/asset_${safe_asset}"
  echo "[RUN] Baselines vs best chain for asset=${asset}"
  env PYTHONPATH=src .venv/bin/python scripts/evaluate_regime_baselines.py \
    --horizons "${HORIZONS[@]}" \
    --tickers "$asset" \
    --walkforward_root "$WF_ROOT" \
    --xgb_n_jobs "$XGB_N_JOBS" \
    --outdir "$asset_out"
done

env OUTDIR="$OUTDIR" .venv/bin/python - <<'PY'
import os
from pathlib import Path
import pandas as pd

outdir = Path(os.environ["OUTDIR"])
assets = [("^GSPC", "_GSPC"), ("BTC-USD", "BTC_USD"), ("TLT", "TLT")]

parts = {
    "metrics_all_splits.csv": [],
    "comparison_test.csv": [],
    "improvements_vs_baselines.csv": [],
    "class_balance.csv": [],
}

for asset, safe in assets:
    asset_dir = outdir / f"asset_{safe}"
    for fname in parts:
        path = asset_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.insert(0, "asset", asset)
        parts[fname].append(df)

for fname, frames in parts.items():
    if not frames:
        continue
    out = pd.concat(frames, ignore_index=True)
    if fname == "comparison_test.csv":
        out = out.sort_values(["asset", "horizon", "macro_f1"], ascending=[True, True, False])
        out_name = "comparison_test_all_assets.csv"
    elif fname == "metrics_all_splits.csv":
        out = out.sort_values(["asset", "horizon", "split", "model"])
        out_name = "metrics_all_splits_all_assets.csv"
    elif fname == "improvements_vs_baselines.csv":
        out = out.sort_values(["asset", "horizon", "delta_macro_f1_vs_persistence"], ascending=[True, True, False])
        out_name = "improvements_vs_baselines_all_assets.csv"
    elif fname == "class_balance.csv":
        out = out.sort_values(["asset", "horizon", "split"])
        out_name = "class_balance_all_assets.csv"
    else:
        out_name = f"all_assets_{fname}"
    out.to_csv(outdir / out_name, index=False)

report = outdir / "report_best_chain_vs_baselines.md"
cmp_path = outdir / "comparison_test_all_assets.csv"
imp_path = outdir / "improvements_vs_baselines_all_assets.csv"
with report.open("w", encoding="utf-8") as f:
    f.write("# Baselines vs Best Chain (from Walk-Forward)\n\n")
    f.write(f"- Output: `{outdir}`\n")
    if cmp_path.exists():
        cmp_df = pd.read_csv(cmp_path)
        f.write("\n## Test Metrics\n\n")
        cols = [c for c in ["asset", "model", "horizon", "accuracy", "macro_f1", "weighted_f1", "n_eval"] if c in cmp_df.columns]
        f.write(cmp_df[cols].to_string(index=False))
        f.write("\n")
    if imp_path.exists():
        imp_df = pd.read_csv(imp_path)
        f.write("\n## Improvements vs Baselines\n\n")
        cols = [
            c for c in [
                "asset",
                "model",
                "horizon",
                "delta_acc_vs_majority",
                "delta_macro_f1_vs_majority",
                "delta_weighted_f1_vs_majority",
                "delta_acc_vs_persistence",
                "delta_macro_f1_vs_persistence",
                "delta_weighted_f1_vs_persistence",
            ]
            if c in imp_df.columns
        ]
        f.write(imp_df[cols].to_string(index=False))
        f.write("\n")
print(f"[OK] Consolidated files written to: {outdir}")
PY

echo "[DONE] Completed baseline comparison vs walk-forward best chain."
