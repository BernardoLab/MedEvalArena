#!/usr/bin/env bash
set -euo pipefail

# Aggregate QuizBench results using judge-valid filters
# (min_med_score=1, require_logical_valid=True, logical_mode=majority),
# but only keep answer models ("quiztakers")
# that have eval outputs for *every* quiz in the batch.
#
# Usage:
#   bash scripts/aggregate_filtered_results_alltakers.sh ABMS20260101
# or:
#   QUIZ_BATCH_TAG=ABMS20260101 bash scripts/aggregate_filtered_results_alltakers.sh
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

OUT_CSV="${OUT_CSV:-/tmp/agg_majority_alltakers${tag_suffix}.csv}"
OUT_CSV_BY_MODEL="${OUT_CSV_BY_MODEL:-/tmp/agg_majority_by_model_alltakers${tag_suffix}.csv}"
OUT_HEATMAP="${OUT_HEATMAP:-filtered_majority_alltakers${tag_suffix}.png}"

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
  --filter_min_med_score 1
  --filter_require_logical_valid
  --filter_logical_mode majority
  --require_complete_answer_models
  --no-restrict_answer_models_to_generators
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
