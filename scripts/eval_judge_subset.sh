#!/usr/bin/env bash
set -euo pipefail

# Optional: select a specific quiz batch tag (e.g., Jan2026).
# You can pass it as the first positional arg:
#   bash scripts/eval_judge_subset.sh Jan2026
# or via env var:
#   QUIZ_BATCH_TAG=Jan2026 bash scripts/eval_judge_subset.sh
QUIZ_BATCH_TAG="${QUIZ_BATCH_TAG:-${1:-}}"

# Required: CSV list of runs roots. Judge models are optional (default list used).
RUNS_ROOTS_CSV="${RUNS_ROOTS_CSV:-}"
JUDGE_MODELS_CSV="${JUDGE_MODELS_CSV:-}"

# Optional: only judge specific quiz_ids and/or generator models.
ONLY_QUIZ_IDS_CSV="${ONLY_QUIZ_IDS_CSV:-}"
ONLY_GENERATOR_MODELS_CSV="${ONLY_GENERATOR_MODELS_CSV:-}"

# Optional: if set and the directory exists, rewrite any runs_root entries that
# start with `eval_results/quizbench/runs/` to point at:
#   ${BATCH_RUNS_ROOT_BASE}/<generator_dir>
# Default: eval_results/quizbench/quizzes_${QUIZ_BATCH_TAG}/runs
BATCH_RUNS_ROOT_BASE="${BATCH_RUNS_ROOT_BASE:-}"
REWRITE_BATCH_RUNS_ROOTS="${REWRITE_BATCH_RUNS_ROOTS:-1}"

# If QUIZ_BATCH_TAG is set, require a matching manifest under each runs_root.
# Set to 0 to skip runs_roots that don't have the batch manifest.
STRICT_BATCH_MANIFEST="${STRICT_BATCH_MANIFEST:-1}"

# Toggle batch API (set to empty string to disable)
# BATCH_FLAG="--use_batch_api"
BATCH_FLAG="${BATCH_FLAG:-}"

# Optional caps and knobs.
MAX_TOKENS="${MAX_TOKENS:-16000}"
MAX_QUIZZES="${MAX_QUIZZES:-0}"

# Debug: set to 1 to print commands without executing.
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat << 'EOF'
Usage:
  RUNS_ROOTS_CSV="eval_results/quizbench/quizzes_Jan2026/runs/claude-opus-4-5-20251101" \
  JUDGE_MODELS_CSV="gemini-3-pro-preview,deepseek-v3.2" \
  ONLY_QUIZ_IDS_CSV="20251219T234603131Z_kimi-k2-thinking_seed129,..." \
  QUIZ_BATCH_TAG=Jan2026 \
  bash scripts/eval_judge_subset.sh

Examples:
  RUNS_ROOTS_CSV="eval_results/quizbench/quizzes_Jan2026/runs/kimi-k2-thinking" \
  JUDGE_MODELS_CSV="gemini-3-pro-preview,claude-opus-4-5-20251101" \
  ONLY_QUIZ_IDS_CSV="quiz_id_1,quiz_id_2" \
  bash scripts/eval_judge_subset.sh Jan2026

Env vars:
  RUNS_ROOTS_CSV            CSV list of runs_root paths (required)
  JUDGE_MODELS_CSV          CSV list of judge models (optional; defaults used if empty)
  ONLY_QUIZ_IDS_CSV         CSV list of quiz_ids to judge
  ONLY_GENERATOR_MODELS_CSV CSV list of generator_models to include
  QUIZ_BATCH_TAG            Same as positional BATCH_TAG (e.g., Jan2026)
  BATCH_RUNS_ROOT_BASE      Override batch base runs dir (default: eval_results/quizbench/quizzes_<BATCH_TAG>/runs)
  REWRITE_BATCH_RUNS_ROOTS  1/0 (default: 1)
  STRICT_BATCH_MANIFEST     1/0 (default: 1)
  MAX_TOKENS                Per-call max tokens (default: 16000)
  MAX_QUIZZES               Optional cap (0 means no cap)
  BATCH_FLAG                Set to --use_batch_api to enable batch judge calls
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

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

split_csv() {
  local csv="$1"
  local IFS=","
  read -r -a parts <<< "$csv"
  for part in "${parts[@]}"; do
    part="$(trim "$part")"
    if [[ -n "$part" ]]; then
      echo "$part"
    fi
  done
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

if [[ -z "${RUNS_ROOTS_CSV}" ]]; then
  die "RUNS_ROOTS_CSV is required."
fi
# if [[ -z "${JUDGE_MODELS_CSV}" ]]; then
#   die "JUDGE_MODELS_CSV is required."
# fi

RUNS_ROOTS=()
while IFS= read -r item; do
  RUNS_ROOTS+=("$item")
done < <(split_csv "${RUNS_ROOTS_CSV}")

if [[ -z "${JUDGE_MODELS_CSV}" ]]; then
  JUDGE_MODELS=(
    "gpt-5.1-2025-11-13"
    "gemini-3-pro-preview"
    "claude-opus-4-5-20251101"
    "grok-4-0709"
    "deepseek-v3.2"
    "kimi-k2-thinking"
  )
else
  JUDGE_MODELS=()
  while IFS= read -r item; do
    JUDGE_MODELS+=("$item")
  done < <(split_csv "${JUDGE_MODELS_CSV}")
fi

if [[ "${#RUNS_ROOTS[@]}" -eq 0 ]]; then
  die "RUNS_ROOTS_CSV resolved to no runs_root entries."
fi
if [[ "${#JUDGE_MODELS[@]}" -eq 0 ]]; then
  die "JUDGE_MODELS_CSV resolved to no judge models."
fi

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
      --max_tokens "${MAX_TOKENS}"
    )
    if [[ -n "${MANIFEST_PATH}" ]]; then
      cmd+=(--manifest_path "${MANIFEST_PATH}")
    fi
    if [[ -n "${ONLY_QUIZ_IDS_CSV}" ]]; then
      cmd+=(--only_quiz_ids_csv "${ONLY_QUIZ_IDS_CSV}")
    fi
    if [[ -n "${ONLY_GENERATOR_MODELS_CSV}" ]]; then
      cmd+=(--only_generator_models_csv "${ONLY_GENERATOR_MODELS_CSV}")
    fi
    if [[ "${MAX_QUIZZES}" =~ ^[0-9]+$ && "${MAX_QUIZZES}" -gt 0 ]]; then
      cmd+=(--max_quizzes "${MAX_QUIZZES}")
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
