#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="runs/walk_forward_chain_xgb_night_${TS}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "$LOG_DIR"

echo "[INFO] Night run root: ${OUT_ROOT}"
echo "[INFO] Logs: ${LOG_DIR}"

run_cmd_with_retry() {
  local name="$1"
  local logfile="$2"
  shift 2

  local attempt=1
  local max_attempts=2
  while true; do
    echo "[RUN] ${name} attempt=${attempt}" | tee -a "$logfile"
    if "$@" >>"$logfile" 2>&1; then
      echo "[OK] ${name}" | tee -a "$logfile"
      return 0
    fi
    if [[ "$attempt" -ge "$max_attempts" ]]; then
      echo "[FAIL] ${name} after ${attempt} attempts" | tee -a "$logfile"
      return 1
    fi
    attempt=$((attempt + 1))
    echo "[WARN] ${name} failed, retrying..." | tee -a "$logfile"
    sleep 5
  done
}

FEATURE_LOG="${LOG_DIR}/00_build_features.log"
run_cmd_with_retry \
  "build_features" \
  "$FEATURE_LOG" \
  env PYTHONPATH=src .venv/bin/python scripts/build_features.py \
    --config_features config/features.yaml \
    --config_sources config/datasources.yaml

assets=("^GSPC" "BTC-USD" "TLT")
horizons=(5 20)
MAX_PARALLEL_ASSETS="${MAX_PARALLEL_ASSETS:-3}"
MAX_STRUCT_CONFIGS="${MAX_STRUCT_CONFIGS:-4}"
MAX_XGB_CONFIGS="${MAX_XGB_CONFIGS:-2}"
PRUNE_AFTER_FOLDS="${PRUNE_AFTER_FOLDS:-2}"
PRUNE_DELTA_THRESHOLD="${PRUNE_DELTA_THRESHOLD:--0.01}"
XGB_N_JOBS="${XGB_N_JOBS:-1}"
MIN_HIGH_VOL_RECALL_DELTA="${MIN_HIGH_VOL_RECALL_DELTA:-0.0}"
BLEND_ALPHAS="${BLEND_ALPHAS:-0.0,0.2,0.4,0.6,0.8,1.0}"

for h in "${horizons[@]}"; do
  echo "[INFO] Starting horizon h=${h} with up to ${MAX_PARALLEL_ASSETS} assets in parallel"
  pids=()
  running=0
  fail_h=0

  for asset in "${assets[@]}"; do
    safe_asset="${asset//[^a-zA-Z0-9]/_}"
    logf="${LOG_DIR}/walk_h${h}_${safe_asset}.log"
    run_cmd_with_retry \
      "walk_forward_h${h}_${asset}" \
      "$logf" \
      env PYTHONPATH=src .venv/bin/python scripts/walk_forward_chain_xgb.py \
        --asset "$asset" \
        --horizon "$h" \
        --grid_profile promising \
        --min_train_end 2018-12-31 \
        --max_valid_end 2023-12-31 \
        --valid_months 12 \
        --step_months 6 \
        --min_folds_ok 2 \
        --min_positive_rate 0.5 \
        --min_high_vol_recall_delta "${MIN_HIGH_VOL_RECALL_DELTA}" \
        --stability_lambda 1.0 \
        --positive_rate_lambda 0.5 \
        --max_struct_configs "${MAX_STRUCT_CONFIGS}" \
        --max_xgb_configs "${MAX_XGB_CONFIGS}" \
        --prune_after_folds "${PRUNE_AFTER_FOLDS}" \
        --prune_delta_threshold "${PRUNE_DELTA_THRESHOLD}" \
        --xgb_n_jobs "${XGB_N_JOBS}" \
        --use_blend \
        --blend_alphas "${BLEND_ALPHAS}" \
        --outdir "$OUT_ROOT" &
    pids+=("$!")
    running=$((running + 1))

    if [[ "$running" -ge "$MAX_PARALLEL_ASSETS" ]]; then
      if ! wait "${pids[0]}"; then
        fail_h=1
      fi
      pids=("${pids[@]:1}")
      running=$((running - 1))
    fi
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail_h=1
    fi
  done
  if [[ "$fail_h" -ne 0 ]]; then
    echo "[FAIL] One or more runs failed in horizon h=${h}"
    exit 1
  fi
done

REPORT_MD="${OUT_ROOT}/report.md"
SUMMARY_PY_LOG="${LOG_DIR}/99_summary.log"

env OUT_ROOT="$OUT_ROOT" PYTHONPATH=src .venv/bin/python - <<'PY' >"$SUMMARY_PY_LOG" 2>&1
import os
from pathlib import Path
import pandas as pd

out_root = Path(os.environ["OUT_ROOT"])
all_rows = []
for f in sorted(out_root.glob("h*/asset_*/final_vs_persistence.csv")):
    try:
        df = pd.read_csv(f)
        if len(df):
            all_rows.append(df.assign(source=str(f)))
    except Exception:
        pass

if all_rows:
    final = pd.concat(all_rows, ignore_index=True)
    final = final.sort_values(["horizon", "asset"])
    final.to_csv(out_root / "final_vs_persistence_all.csv", index=False)

    with open(out_root / "report.md", "w", encoding="utf-8") as rep:
        rep.write("# Walk-Forward Night Report\n\n")
        rep.write(f"- Output root: `{out_root}`\n")
        rep.write("- Scope: per-asset (^GSPC, BTC-USD, TLT), horizons h=5 and h=20\n")
        rep.write("- Selection: robust score (delta vs persistence + stability penalties)\n\n")
        rep.write("## Final vs Persistence (Test)\n\n")
        rep.write(final[[
            "asset",
            "horizon",
            "chain_test_acc",
            "chain_test_macro_f1",
            "chain_test_weighted_f1",
            "persistence_test_acc",
            "persistence_test_macro_f1",
            "persistence_test_weighted_f1",
            "delta_test_acc_vs_persistence",
            "delta_test_macro_f1_vs_persistence",
            "delta_test_weighted_f1_vs_persistence",
            "selected_experiment_id",
            "n_features",
        ]].to_markdown(index=False))
        rep.write("\n\n## Notes\n\n")
        rep.write("- Positive `delta_test_macro_f1_vs_persistence` means chain beats persistence.\n")
        rep.write("- Detailed fold metrics and summaries are under each `h*/asset_*` folder.\n")
else:
    with open(out_root / "report.md", "w", encoding="utf-8") as rep:
        rep.write("# Walk-Forward Night Report\n\n")
        rep.write("- No `final_vs_persistence.csv` files found. Check logs.\n")
PY

echo "[DONE] Night run finished. Report: ${REPORT_MD}"
