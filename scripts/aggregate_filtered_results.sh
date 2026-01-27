#!/usr/bin/env bash
set -euo pipefail

# Optional: aggregate only a specific quiz batch tag (e.g., Jan2026).
# You can pass it as the first positional arg:
#   bash scripts/aggregate_filtered_results.sh Jan2026
# or via env var:
#   QUIZ_BATCH_TAG=Jan2026 bash scripts/aggregate_filtered_results.sh
QUIZ_BATCH_TAG="${QUIZ_BATCH_TAG:-${1:-}}"

# Base runs root (defaults to the legacy location). If QUIZ_BATCH_TAG is set
# and the batch root exists, prefer it.
RUNS_ROOT="${RUNS_ROOT:-eval_results/quizbench/runs}"
if [[ -n "${QUIZ_BATCH_TAG}" ]]; then
  BATCH_ROOT="eval_results/quizbench/quizzes_${QUIZ_BATCH_TAG}/runs"
  if [[ -d "${BATCH_ROOT}" ]]; then
    RUNS_ROOT="${BATCH_ROOT}"
  fi
fi

tag_suffix=""
if [[ -n "${QUIZ_BATCH_TAG}" ]]; then
  tag_suffix="_${QUIZ_BATCH_TAG}"
fi

OUT_CSV="${OUT_CSV:-/tmp/agg_majority${tag_suffix}.csv}"
OUT_CSV_BY_MODEL="${OUT_CSV_BY_MODEL:-/tmp/agg_majority_by_model${tag_suffix}.csv}"
OUT_HEATMAP="${OUT_HEATMAP:-eval_results/quizbench/quizzes${tag_suffix}/filtered_majority${tag_suffix}.png}"
OUT_VALIDITYBARPLOT="${VALIDITYBARPLOT:-eval_results/quizbench/quizzes${tag_suffix}/filtered_validity_barp${tag_suffix}.png}"

# Debug: set to 1 to print the command without executing.
DRY_RUN="${DRY_RUN:-0}"

printf "Running quizbench/aggregate_results.py\n"
printf "runs_root = ${RUNS_ROOT}\n"


cmd=(
  uv run quizbench/aggregate_results.py
  --runs_root "${RUNS_ROOT}"
  --out_csv "${OUT_CSV}"
  --out_csv_by_model "${OUT_CSV_BY_MODEL}"
  --out_heatmap "${OUT_HEATMAP}"
  --out_filtered_barplot "${OUT_VALIDITYBARPLOT}"
  --filter_min_med_score 1
  --filter_require_logical_valid
  --filter_logical_mode majority
)

if [[ -n "${QUIZ_BATCH_TAG}" ]]; then
  cmd+=(--quiz_batch_tag "${QUIZ_BATCH_TAG}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf "[DRY-RUN] "
  printf "%q " "${cmd[@]}"
  printf "\n"
  exit 0
fi

"${cmd[@]}"
