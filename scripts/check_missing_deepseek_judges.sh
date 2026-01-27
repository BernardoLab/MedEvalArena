#!/usr/bin/env bash
set -euo pipefail

# Scan quizbench runs and report quizzes that are missing
# deepseek-v3.2 judge outputs.
#
# Usage:
#   bash scripts/check_missing_deepseek_judges.sh
#
# Optional env vars:
#   JUDGE_MODEL       Judge model name (default: deepseek-v3.2)
#   RUNS_ROOT_BASE    Base directory for runs (default: eval_results/quizbench/runs)

JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v3.2}"
RUNS_ROOT_BASE="${RUNS_ROOT_BASE:-eval_results/quizbench/runs}"

echo "[INFO] Checking for missing judge outputs"
echo "[INFO]   JUDGE_MODEL=${JUDGE_MODEL}"
echo "[INFO]   RUNS_ROOT_BASE=${RUNS_ROOT_BASE}"

if [[ ! -d "${RUNS_ROOT_BASE}" ]]; then
  echo "[ERROR] RUNS_ROOT_BASE does not exist: ${RUNS_ROOT_BASE}" >&2
  exit 1
fi

for runs_root in "${RUNS_ROOT_BASE}"/*; do
  [[ -d "${runs_root}" ]] || continue
  compgen -G "${runs_root}/quizbench_manifest*.json" > /dev/null || continue

  unstarted_csv=""
  partial_csv=""

  for quiz_dir in "${runs_root}"/*seed*; do
    [[ -d "${quiz_dir}" ]] || continue
    quiz_id="$(basename "${quiz_dir}")"

    result_path="${quiz_dir}/${JUDGE_MODEL}_judge_result.json"
    summary_path="${quiz_dir}/${JUDGE_MODEL}_judge_summary.json"

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
  echo "=== ${runs_root} ==="

  if [[ -n "${unstarted_csv}" ]]; then
    echo "[UNSTARTED] quiz_ids_csv:"
    echo "  ${unstarted_csv}"
    echo "  # Example resume:"
    echo "  uv run python -m quizbench.run_eval_judge \\"
    echo "    --runs_root \"${runs_root}\" \\"
    echo "    --judge_model \"${JUDGE_MODEL}\" \\"
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
