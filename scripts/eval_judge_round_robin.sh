#!/usr/bin/env bash
set -euo pipefail

# Optional: select a specific quiz batch tag (e.g., Jan2026).
# You can pass it as the first positional arg:
#   bash scripts/eval_judge_round_robin.sh Jan2026
# or via env var:
#   QUIZ_BATCH_TAG=Jan2026 bash scripts/eval_judge_round_robin.sh
QUIZ_BATCH_TAG="${QUIZ_BATCH_TAG:-${1:-}}"

# Optional: if set and the directory exists, rewrite any runs_root entries that
# start with `eval_results/quizbench/runs/` to point at:
#   ${BATCH_RUNS_ROOT_BASE}/<generator_dir>
# Default: eval_results/quizbench/quizzes_${QUIZ_BATCH_TAG}/runs
BATCH_RUNS_ROOT_BASE="${BATCH_RUNS_ROOT_BASE:-}"
REWRITE_BATCH_RUNS_ROOTS="${REWRITE_BATCH_RUNS_ROOTS:-1}"

# If QUIZ_BATCH_TAG is set, require a matching manifest under each runs_root.
# Set to 0 to skip runs_roots that don't have the batch manifest.
STRICT_BATCH_MANIFEST="${STRICT_BATCH_MANIFEST:-1}"

# Debug: set to 1 to print commands without executing.
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat << 'EOF'
Usage:
  bash scripts/eval_judge_round_robin.sh [BATCH_TAG]

Examples:
  bash scripts/eval_judge_round_robin.sh Jan2026
  QUIZ_BATCH_TAG=Jan2026 bash scripts/eval_judge_round_robin.sh
  DRY_RUN=1 QUIZ_BATCH_TAG=Jan2026 bash scripts/eval_judge_round_robin.sh

Behavior when BATCH_TAG is set:
  - Uses the newest manifest matching quizbench_manifest_<BATCH_TAG>*.json under each runs_root.
  - Optionally rewrites runs_root entries from eval_results/quizbench/runs/<gen>
    to eval_results/quizbench/quizzes_<BATCH_TAG>/runs/<gen> when that base exists.

Env vars:
  QUIZ_BATCH_TAG            Same as positional BATCH_TAG (e.g., Jan2026)
  BATCH_RUNS_ROOT_BASE      Override batch base runs dir (default: eval_results/quizbench/quizzes_<BATCH_TAG>/runs)
  REWRITE_BATCH_RUNS_ROOTS  1/0 (default: 1)
  STRICT_BATCH_MANIFEST     1/0 (default: 1)
  DRY_RUN                   1/0 (default: 0)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

resolve_runs_root() {
  local runs_root="$1"

  if [[ -z "${QUIZ_BATCH_TAG}" || "${REWRITE_BATCH_RUNS_ROOTS}" != "1" ]]; then
    echo "${runs_root}"
    return 0
  fi

  local base="${BATCH_RUNS_ROOT_BASE}"
  if [[ -z "${base}" ]]; then
    base="eval_results/quizbench/quizzes_${QUIZ_BATCH_TAG}/runs"
  fi
  if [[ ! -d "${base}" ]]; then
    echo "${runs_root}"
    return 0
  fi

  if [[ "${runs_root}" == "eval_results/quizbench/runs/"* ]]; then
    local suffix="${runs_root#eval_results/quizbench/runs/}"
    local candidate="${base}/${suffix}"
    if [[ -d "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  fi

  echo "${runs_root}"
}

resolve_manifest_for_batch() {
  local runs_root="$1"
  local batch_tag="$2"

  local pattern="quizbench_manifest_${batch_tag}"'*.json'
  if ! compgen -G "${runs_root}/${pattern}" > /dev/null; then
    echo ""
    return 0
  fi

  # Return a path *relative to runs_root* (run_eval_judge resolves relative paths
  # against --runs_root).
  (cd "${runs_root}" && ls -1t ${pattern} 2>/dev/null | head -n 1) || true
}

# Directories containing quizbench manifests + quiz runs to be judged
RUNS_ROOTS=(
  # "eval_results/quizbench/runs/gpt-5.1-2025-11-13" 
  # "eval_results/quizbench/runs/gemini-3-pro-preview"
  "eval_results/quizbench/runs/claude-opus-4-5-20251101" #[ ]
  # "eval_results/quizbench/runs/deepseek-v3.2" # [x]
  # "eval_results/quizbench/runs/kimi-k2-thinking"
  # "eval_results/quizbench/runs/grok-4-0709" # [x]
)

# Judge models to run on every quiz set
JUDGE_MODELS=(
  # "gpt-5.1-2025-11-13" # [x] [x]
  # "gemini-3-pro-preview" # [ ] (ran out of tokens!)
  # "claude-opus-4-5-20251101" # [x] 
  # "grok-4-0709" # 
  # "deepseek-v3.2" # 
  "kimi-k2-thinking" # 
)

# Toggle batch API (set to empty string to disable)
# BATCH_FLAG="--use_batch_api"
BATCH_FLAG=""

# Debug: set MAX_QUIZZES to 1 to debug; else 0
MAX_QUIZZES=0
# MAX_QUIZZES=1

for MODEL in "${JUDGE_MODELS[@]}"; do
  echo "[INFO] === Judge model: ${MODEL} ==="
  for RUNS_ROOT in "${RUNS_ROOTS[@]}"; do
    RESOLVED_RUNS_ROOT="$(resolve_runs_root "${RUNS_ROOT}")"
    echo "[INFO]   runs_root: ${RESOLVED_RUNS_ROOT}"

    MANIFEST_PATH=""
    if [[ -n "${QUIZ_BATCH_TAG}" ]]; then
      MANIFEST_PATH="$(resolve_manifest_for_batch "${RESOLVED_RUNS_ROOT}" "${QUIZ_BATCH_TAG}")"
      if [[ -z "${MANIFEST_PATH}" ]]; then
        msg="No manifest found under ${RESOLVED_RUNS_ROOT} matching quizbench_manifest_${QUIZ_BATCH_TAG}*.json"
        if [[ "${STRICT_BATCH_MANIFEST}" == "1" ]]; then
          die "${msg}"
        fi
        echo "[WARN] ${msg}; skipping."
        continue
      fi
      echo "[INFO]     manifest_path: ${RESOLVED_RUNS_ROOT}/${MANIFEST_PATH}"
    fi

    cmd=(
      uv run python -m quizbench.run_eval_judge
      --runs_root "${RESOLVED_RUNS_ROOT}"
      --judge_model "${MODEL}"
      --max_tokens 16000
      --max_quizzes "${MAX_QUIZZES}"
    )
    if [[ -n "${MANIFEST_PATH}" ]]; then
      cmd+=(--manifest_path "${MANIFEST_PATH}")
    fi
    if [[ -n "${BATCH_FLAG}" ]]; then
      cmd+=("${BATCH_FLAG}")
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
