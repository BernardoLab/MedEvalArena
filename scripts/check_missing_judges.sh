#!/usr/bin/env bash
set -euo pipefail

# Scan quizbench runs and report quizzes that are missing
# judge outputs.
#
# Usage:
#   bash scripts/check_missing_judges.sh
#
# Optional env vars:
#   JUDGE_MODELS_CSV  Comma-separated judge model names
#   JUDGE_MODEL       Single judge model name (overrides list)
#   RUNS_ROOT_BASE    Base directory for runs (default: eval_results/quizbench/runs)
#   DEEPER_MAX_UNITS  Max units to list in deep scan (0 disables listing; default: 50)

# Edit this list to check multiple judges by default.
JUDGE_MODELS_DEFAULT=(
  "gpt-5.1-2025-11-13"
  "gemini-3-pro-preview"
  "claude-opus-4-5-20251101"
  "grok-4-0709"
  "deepseek-v3.2"
  "kimi-k2-thinking"
)

JUDGE_MODELS=("${JUDGE_MODELS_DEFAULT[@]}")
if [[ -n "${JUDGE_MODELS_CSV:-}" ]]; then
  IFS=',' read -r -a JUDGE_MODELS <<< "${JUDGE_MODELS_CSV}"
fi
if [[ -n "${JUDGE_MODEL:-}" ]]; then
  JUDGE_MODELS=("${JUDGE_MODEL}")
fi

RUNS_ROOT_BASE="${RUNS_ROOT_BASE:-eval_results/quizbench/quizzes_Jan2026/runs}"
DEEPER_MAX_UNITS="${DEEPER_MAX_UNITS:-50}"

echo "[INFO] Checking for missing judge outputs"
echo "[INFO]   JUDGE_MODELS=${JUDGE_MODELS[*]}"
echo "[INFO]   RUNS_ROOT_BASE=${RUNS_ROOT_BASE}"

if [[ "${#JUDGE_MODELS[@]}" -eq 0 ]]; then
  echo "[ERROR] No judge models configured. Set JUDGE_MODEL or JUDGE_MODELS_CSV." >&2
  exit 1
fi

if [[ ! -d "${RUNS_ROOT_BASE}" ]]; then
  echo "[ERROR] RUNS_ROOT_BASE does not exist: ${RUNS_ROOT_BASE}" >&2
  exit 1
fi

for runs_root in "${RUNS_ROOT_BASE}"/*; do
  [[ -d "${runs_root}" ]] || continue
  compgen -G "${runs_root}/quizbench_manifest*.json" > /dev/null || continue

  echo
  echo "=== ${runs_root} ==="

  for judge_model in "${JUDGE_MODELS[@]}"; do
    unstarted_csv=""
    partial_csv=""

    for quiz_dir in "${runs_root}"/*seed*; do
      [[ -d "${quiz_dir}" ]] || continue
      quiz_id="$(basename "${quiz_dir}")"

      result_path="${quiz_dir}/${judge_model}_judge_result.json"
      summary_path="${quiz_dir}/${judge_model}_judge_summary.json"

      if [[ ! -f "${result_path}" && ! -f "${summary_path}" ]]; then
        if [[ -z "${unstarted_csv}" ]]; then
          unstarted_csv="${quiz_id}"
        else
          unstarted_csv="${unstarted_csv},${quiz_id}"
        fi
      elif [[ ! -f "${result_path}" || ! -f "${summary_path}" ]]; then
        if [[ -z "${partial_csv}" ]]; then
          partial_csv="${quiz_id}"
        else
          partial_csv="${partial_csv},${quiz_id}"
        fi
      fi
    done

    echo
    echo "[JUDGE] ${judge_model}"

    if [[ -n "${unstarted_csv}" ]]; then
      echo "[UNSTARTED] quiz_ids_csv:"
      echo "  ${unstarted_csv}"
      echo "  # Example resume:"
      echo "  uv run python -m quizbench.run_eval_judge \\"
      echo "    --runs_root \"${runs_root}\" \\"
      echo "    --judge_model \"${judge_model}\" \\"
      echo "    --only_quiz_ids_csv \"${unstarted_csv}\" \\"
      echo "    --max_tokens 16000"
    else
      echo "[UNSTARTED] none"
    fi

    if [[ -n "${partial_csv}" ]]; then
      echo "[PARTIAL] quiz_ids_csv (result/summary mismatch):"
      echo "  ${partial_csv}"
    else
      echo "[PARTIAL] none"
    fi
  done

  echo
  python3 quizbench/check_missing_judges_deep.py \
    --runs_root "${runs_root}" \
    --judge_models "${JUDGE_MODELS[@]}" \
    --max_units "${DEEPER_MAX_UNITS}"
done
