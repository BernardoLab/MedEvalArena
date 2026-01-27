#!/usr/bin/env bash
set -euo pipefail

# Evaluate a date-tagged ABMS subset (e.g., ABMS20260101) in a round-robin loop.
#
# This expects the subset to have been built by:
#   uv run python -m quizbench.build_abms_valid_subset --subset_tag ABMS20260101 ...
#
# Default runs root base:
#   eval_results/quizbench/quizzes_ABMS20260101/runs/<generator_model>/
#
# Usage:
#   bash scripts/eval_round_robin_subset.sh ABMS20260101
#   SUBSET_TAG=ABMS20260101 bash scripts/eval_round_robin_subset.sh
#
# Optional env vars:
#   RUNS_ROOT_BASE     Override base runs root (default uses SUBSET_TAG)
#   GEN_RUNS_CSV       Comma-separated generator runs_root paths (overrides auto-discovery)
#   EVAL_MODELS_CSV    Comma-separated eval models (overrides the hardcoded list below)
#   USE_BATCH_API      1/0 (default: 1)
#   MAX_TOKENS         Passed through to run_eval.py (default: 4000)
#   REASONING_EFFORT   Passed through to run_eval.py (default: high)
#   CONFIG_PATH        Optional YAML config passed to run_eval.py
#   DRY_RUN           1/0 (default: 0)

SUBSET_TAG="${SUBSET_TAG:-${1:-}}"
if [[ "${SUBSET_TAG}" == "-h" || "${SUBSET_TAG}" == "--help" || -z "${SUBSET_TAG}" ]]; then
  cat << 'EOF'
Usage:
  bash scripts/eval_round_robin_subset.sh <SUBSET_TAG>

Example:
  bash scripts/eval_round_robin_subset.sh ABMS20260101
EOF
  exit 0
fi

RUNS_ROOT_BASE="${RUNS_ROOT_BASE:-eval_results/quizbench/quizzes_${SUBSET_TAG}/runs}"
GEN_RUNS_CSV="${GEN_RUNS_CSV:-}"
EVAL_MODELS_CSV="${EVAL_MODELS_CSV:-}"
USE_BATCH_API="${USE_BATCH_API:-1}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
CONFIG_PATH="${CONFIG_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

[[ -d "${RUNS_ROOT_BASE}" ]] || die "RUNS_ROOT_BASE does not exist: ${RUNS_ROOT_BASE}"

# Discover generator runs roots.
GEN_RUNS=()
if [[ -n "${GEN_RUNS_CSV}" ]]; then
  IFS=',' read -r -a GEN_RUNS <<< "${GEN_RUNS_CSV}"
else
  while IFS= read -r -d '' d; do
    GEN_RUNS+=("${d}")
  done < <(find "${RUNS_ROOT_BASE}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

[[ "${#GEN_RUNS[@]}" -gt 0 ]] || die "No generator directories found under: ${RUNS_ROOT_BASE}"

# Eval models.
EVAL_MODELS=(
"grok-4-1-fast-reasoning"
#  "gemini-2.5-flash-lite"
#  "claude-haiku-4-5-20251001"
#  "gpt-5-nano-2025-08-07"
#  "gpt-5.1-2025-11-13"
#  "gemini-3-pro-preview"
#  "claude-opus-4-5-20251101"
#  "grok-4-0709"
#  "kimi-k2-thinking"
#  "deepseek-v3.2"
)
if [[ -n "${EVAL_MODELS_CSV}" ]]; then
  IFS=',' read -r -a EVAL_MODELS <<< "${EVAL_MODELS_CSV}"
fi

MANIFEST_BASENAME="quizbench_manifest_${SUBSET_TAG}.json"

for MODEL in "${EVAL_MODELS[@]}"; do
  echo "[INFO] === Eval model: ${MODEL} ==="
  for RUNS_ROOT in "${GEN_RUNS[@]}"; do
    echo "[INFO]   runs_root: ${RUNS_ROOT}"

    cmd=(uv run python -m quizbench.run_eval
      --runs_root "${RUNS_ROOT}"
      --manifest_path "${MANIFEST_BASENAME}"
      --eval_model "${MODEL}"
      --max_tokens "${MAX_TOKENS}"
      --reasoning_effort "${REASONING_EFFORT}"
    )

    if [[ -n "${CONFIG_PATH}" ]]; then
      cmd+=(--config "${CONFIG_PATH}")
    fi
    if [[ "${USE_BATCH_API}" == "1" ]]; then
      cmd+=(--use_batch_api)
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf "[DRY-RUN] "
      printf "%q " "${cmd[@]}"
      printf "\n"
      continue
    fi

    "${cmd[@]}"
  done
done

 
